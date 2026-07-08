"""Tests for the deploy coordinator's cross-operator mutex actor factory.

The routing-registry concern lives on a separate actor (see
tests/test_replica_coordinator.py); this file only covers what's left on
DeployCoordinator. The reserve/release/liveness paths currently have no unit
coverage."""

from unittest.mock import MagicMock, patch

from modelship.infer import deploy_coordinator


def test_get_or_create_sets_max_restarts():
    # Resurrection only helps because the actor auto-restarts; assert the option.
    with (
        patch.object(deploy_coordinator.ray, "get_actor", side_effect=ValueError("absent")),
        patch.object(deploy_coordinator.DeployCoordinator, "options") as options,
    ):
        options.return_value.remote.return_value = MagicMock()
        deploy_coordinator.get_or_create_coordinator()
    assert options.call_args.kwargs["max_restarts"] == -1
