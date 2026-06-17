"""Tests for the deploy coordinator's routing registry + watch API (the source of
truth gateway replicas reconcile from). Exercises the undecorated class directly,
in-process, without a Ray cluster."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from modelship.infer import deploy_coordinator
from modelship.infer.deploy_coordinator import ModelshipDeployCoordinator
from modelship.state import MemoryStateStore

# The plain class behind @ray.remote — its async methods are ordinary coroutines.
_Coord = ModelshipDeployCoordinator.__ray_metadata__.modified_class


@pytest.fixture
def coord():
    return _Coord()


class TestRoutingRegistry:
    @pytest.mark.asyncio
    async def test_register_records_and_bumps_generation(self, coord):
        assert (await coord.get_routing("gw"))["generation"] == 0
        await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {"qwen-aaaa": "qwen"}
        assert routing["generation"] == 1

    @pytest.mark.asyncio
    async def test_unregister_removes_and_bumps(self, coord):
        await coord.register_deployment("gw", "qwen-aaaa", "qwen")
        await coord.unregister_deployment("gw", "qwen-aaaa")
        routing = await coord.get_routing("gw")
        assert routing["models"] == {}
        assert routing["generation"] == 2

    @pytest.mark.asyncio
    async def test_set_expected_records_and_bumps(self, coord):
        await coord.set_expected("gw", ["qwen", "kokoro"])
        routing = await coord.get_routing("gw")
        assert routing["expected"] == ["qwen", "kokoro"]
        assert routing["generation"] == 1

    @pytest.mark.asyncio
    async def test_generation_is_per_gateway(self, coord):
        await coord.register_deployment("gw-a", "x-1", "x")
        assert (await coord.get_routing("gw-a"))["generation"] == 1
        assert (await coord.get_routing("gw-b"))["generation"] == 0


class TestWaitForChange:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_advanced(self, coord):
        await coord.register_deployment("gw", "x-1", "x")  # gen -> 1
        # A caller still at gen 0 must not block — the set already moved.
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=1)
        assert gen == 1

    @pytest.mark.asyncio
    async def test_blocks_then_wakes_on_change(self, coord):
        async def mutate():
            await asyncio.sleep(0.05)
            await coord.register_deployment("gw", "x-1", "x")

        task = asyncio.create_task(mutate())
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=2)
        await task
        assert gen == 1

    @pytest.mark.asyncio
    async def test_times_out_returning_same_generation(self, coord):
        gen = await coord.wait_for_change("gw", 0, timeout=0.05)
        assert gen == 0

    @pytest.mark.asyncio
    async def test_restart_lower_generation_returns_immediately(self, coord):
        # Replica last saw gen 5; a restarted coordinator is back at gen 0. Returning
        # at once (0 != 5) lets the replica re-sync instead of blocking forever.
        gen = await asyncio.wait_for(coord.wait_for_change("gw", 5), timeout=1)
        assert gen == 0

    @pytest.mark.asyncio
    async def test_consecutive_cycles_advance(self, coord):
        await coord.register_deployment("gw", "x-1", "x")
        g1 = await asyncio.wait_for(coord.wait_for_change("gw", 0), timeout=1)
        assert g1 == 1
        # Subscribe at g1; a second change wakes us with the next generation.
        waiter = asyncio.create_task(asyncio.wait_for(coord.wait_for_change("gw", g1), timeout=2))
        await asyncio.sleep(0.05)
        await coord.set_expected("gw", ["x"])
        assert await waiter == 2


class TestDurableState:
    """Registry + expected are written through a StateStore so a resurrected
    coordinator (max_restarts) reloads them instead of coming back empty."""

    @pytest.mark.asyncio
    async def test_reloads_registry_and_expected_from_shared_store(self):
        store = MemoryStateStore()  # stand-in for a redis:// store across restarts
        with patch.object(deploy_coordinator, "get_state_store", return_value=store):
            first = _Coord()
            await first.register_deployment("gw", "qwen-aaaa", "qwen")
            await first.set_expected("gw", ["qwen", "kokoro"])

            # A brand-new coordinator backed by the same store = a resurrected actor.
            second = _Coord()
        routing = await second.get_routing("gw")
        assert routing["models"] == {"qwen-aaaa": "qwen"}
        assert routing["expected"] == ["qwen", "kokoro"]
        # Generation is ephemeral — restart resets it to 0 (the gateway treats that
        # as "changed" and re-pulls).
        assert routing["generation"] == 0

    @pytest.mark.asyncio
    async def test_unregister_persists_removal(self):
        store = MemoryStateStore()
        with patch.object(deploy_coordinator, "get_state_store", return_value=store):
            first = _Coord()
            await first.register_deployment("gw", "qwen-aaaa", "qwen")
            await first.unregister_deployment("gw", "qwen-aaaa")
            second = _Coord()
        assert (await second.get_routing("gw"))["models"] == {}

    def test_get_or_create_sets_max_restarts(self):
        # Resurrection only helps because the actor auto-restarts; assert the option.
        with (
            patch.object(deploy_coordinator.ray, "get_actor", side_effect=ValueError("absent")),
            patch.object(deploy_coordinator.ModelshipDeployCoordinator, "options") as options,
        ):
            options.return_value.remote.return_value = MagicMock()
            deploy_coordinator.get_or_create_coordinator()
        assert options.call_args.kwargs["max_restarts"] == -1
