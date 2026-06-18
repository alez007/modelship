"""Tests for ModelshipAPI model discovery and routing."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.openai.api import ModelshipAPI

# Access the underlying class, bypassing the @serve.deployment wrapper.
_ModelshipAPI = ModelshipAPI.func_or_class


@pytest.fixture
def api():
    """Create a ModelshipAPI instance with mocked Ray Serve context. The watch
    loop is marked started so `_ensure_watching` is a no-op — tests drive routing
    directly via `_apply` / `_apply_snapshot`; watch-specific tests reset it."""
    with patch("modelship.openai.api.serve.get_replica_context") as mock_ctx:
        mock_ctx.return_value.app_name = "test-gateway"
        inst = _ModelshipAPI("test-gateway")
        inst._watch_task = MagicMock()
        return inst


def _apply(api, models, *, expected=None, gen=1, handles=None):
    """Apply a coordinator routing snapshot with Serve mocked — the new entry point
    that replaced the driver's add_models/remove pushes.

    - `models`: {app_name: model_name} desired routing.
    - `gen`: snapshot generation. Removals are honored only when it advances; pass
      a lower value than the replica's current `_gen` to simulate a coordinator
      restart (removals suppressed).
    - `handles`: side_effect for serve.get_app_handle (default: a fresh handle per
      call); pass a list to control which handle each app gets.
    """
    with ExitStack() as stack:
        if handles is not None:
            stack.enter_context(patch("modelship.openai.api.serve.get_app_handle", side_effect=handles))
        else:
            stack.enter_context(
                patch("modelship.openai.api.serve.get_app_handle", side_effect=lambda *a, **k: MagicMock())
            )
        api._apply_snapshot({"models": models, "expected": expected or [], "generation": gen})


class TestApplyRouting:
    """The reconcile core the watch loop runs: build/extend the routing table from
    a coordinator snapshot."""

    def test_add_single_model(self, api):
        _apply(api, {"qwen-a3f9k": "qwen"})
        assert "qwen" in api.models
        assert len(api.models["qwen"]) == 1
        assert api.model_list[0].id == "qwen"

    def test_add_multiple_deployments_same_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, handles=[h1, h2])
        assert len(api.models["qwen"]) == 2
        assert len(api.model_list) == 1

    def test_add_different_models(self, api):
        _apply(api, {"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"})
        assert "qwen" in api.models
        assert "kokoro" in api.models
        assert len(api.model_list) == 2

    def test_incremental_snapshot_adds_new_handle_to_existing_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen"}, handles=[h1])
        # A later snapshot adds a 2nd deployment of qwen; the already-routed one is
        # left untouched (no new handle fetched for it).
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, gen=2, handles=[h2])
        assert len(api.models["qwen"]) == 2
        assert api.models["qwen"]["qwen-a3f9k"] is h1
        assert api.models["qwen"]["qwen-b7x2p"] is h2
        assert len(api.model_list) == 1

    def test_handle_failure_skips(self, api):
        _apply(api, {"qwen-a3f9k": "qwen"}, handles=Exception("not found"))
        assert "qwen" not in api.models
        assert len(api.model_list) == 0

    def test_records_per_model_load_times_and_ready_timestamp(self, api):
        _apply(api, {"qwen-a3f9k": "qwen"}, expected=["qwen", "kokoro"])
        assert api._expected_set_at is not None
        assert "qwen" in api._model_load_times
        assert api._model_load_times["qwen"] >= 0
        assert api._all_ready_at is None  # kokoro still pending

        _apply(api, {"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"}, expected=["qwen", "kokoro"], gen=2)
        assert "kokoro" in api._model_load_times
        assert api._all_ready_at is not None

    def test_readyz_body_ready_flag(self, api):
        _apply(api, {}, expected=["qwen"])
        body = api._readyz_body()
        assert body["ready"] is False
        assert body["models_pending"] == ["qwen"]
        assert body["time_to_ready_s"] is None

        _apply(api, {"qwen-a3f9k": "qwen"}, expected=["qwen"], gen=2)
        body = api._readyz_body()
        assert body["ready"] is True
        assert body["models_pending"] == []
        assert body["time_to_ready_s"] is not None
        assert "qwen" in body["model_load_times_s"]


class TestReconcileRemovals:
    """A snapshot that drops an app removes it when the generation advances; a
    regressed generation (coordinator restart) never blanks live routing."""

    def test_dropped_app_removed_on_forward_snapshot(self, api):
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, gen=1)
        assert "qwen" in api.models
        _apply(api, {}, gen=2)
        assert "qwen" not in api.models
        assert api.model_list == []
        assert "qwen" not in api._round_robin

    def test_one_of_many_dropped_keeps_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        _apply(api, {"qwen-aaaaaaaaaa": "qwen", "qwen-bbbbbbbbbb": "qwen"}, gen=1, handles=[h1, h2])
        _apply(api, {"qwen-bbbbbbbbbb": "qwen"}, gen=2)
        assert "qwen" in api.models
        assert list(api.models["qwen"].keys()) == ["qwen-bbbbbbbbbb"]
        assert len(api.model_list) == 1

    def test_regressed_generation_does_not_blank_routing(self, api):
        # Coordinator restarted (generation reset below ours) but the model is still
        # deployed: additions are adopted, live routing is never removed.
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, gen=5)
        _apply(api, {}, gen=0)
        assert "qwen" in api.models

    def test_drop_unknown_app_is_noop(self, api):
        assert api._drop_apps(["nonexistent-1234567890"]) == []

    def test_removal_drops_from_expected_when_snapshot_drops_it(self, api):
        _apply(api, {"qwen-a3f9k1b2c4": "qwen"}, expected=["qwen", "kokoro"], gen=1)
        _apply(api, {}, expected=["kokoro"], gen=2)
        assert api.expected_models == ["kokoro"]


class TestListDeployments:
    @pytest.mark.asyncio
    async def test_returns_app_names_per_model(self, api):
        _apply(
            api,
            {"qwen-aaaaaaaaaa": "qwen", "qwen-bbbbbbbbbb": "qwen", "kokoro-cccccccccc": "kokoro"},
        )
        listed = await api.list_deployments()
        assert set(listed["qwen"]) == {"qwen-aaaaaaaaaa", "qwen-bbbbbbbbbb"}
        assert listed["kokoro"] == ["kokoro-cccccccccc"]


class TestWatchReconcile:
    """The first-request synchronous sync that seeds a (re)started replica from the
    coordinator before the watch loop takes over."""

    def test_sync_pulls_snapshot_and_builds_table(self, api):
        api._watch_task = None  # exercise the real sync path
        snapshot = {
            "models": {"qwen-aaaaaaaaaa": "qwen", "embed-bbbbbbbbbb": "embed"},
            "expected": ["qwen", "embed"],
            "generation": 3,
        }
        with (
            patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=snapshot),
            patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()),
        ):
            assert api._sync_routing_blocking() is True

        assert set(api.models) == {"qwen", "embed"}
        assert api._gen == 3
        assert api._readyz_body()["ready"] is True

    def test_sync_tolerates_unavailable_coordinator(self, api):
        api._watch_task = None
        with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", side_effect=RuntimeError):
            assert api._sync_routing_blocking() is False
        assert api.models == {}

    def test_failed_sync_drops_stale_coordinator_handle(self, api):
        # A cached handle whose actor died (recreated with a new ActorID) must be
        # cleared so the next _coord() re-resolves instead of retrying a corpse.
        stale = MagicMock()
        stale.get_routing.remote.side_effect = RuntimeError("actor dead")
        api._coordinator = stale
        with patch("modelship.openai.api.ray.get", side_effect=RuntimeError("actor dead")):
            assert api._sync_routing_blocking() is False
        assert api._coordinator is None

    @pytest.mark.asyncio
    async def test_coord_async_resolves_off_thread_and_caches(self, api):
        # The watch loop resolves the coordinator via asyncio.to_thread (so the sync
        # ray.get_actor never blocks the event loop) and caches the handle.
        api._coordinator = None
        sentinel = MagicMock()
        with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=sentinel) as goc:
            assert await api._coord_async() is sentinel
            assert await api._coord_async() is sentinel
        goc.assert_called_once()  # second call served from cache, no re-resolve

    def test_sync_keeps_live_models_on_regressed_generation(self, api):
        # Coordinator restarted: empty snapshot at a lower generation than ours. The
        # model is still deployed, so routing is preserved, not blanked.
        _apply(api, {"qwen-aaaaaaaaaa": "qwen"}, gen=4)
        empty = {"models": {}, "expected": [], "generation": 0}
        with (
            patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=empty),
        ):
            api._coordinator = None
            assert api._sync_routing_blocking() is True
        assert "qwen" in api.models


class TestGetHandle:
    def test_round_robin(self, api):
        ha, hb = MagicMock(), MagicMock()
        _apply(api, {"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"}, handles=[ha, hb])
        assert api._get_handle("qwen") is ha
        assert api._get_handle("qwen") is hb
        assert api._get_handle("qwen") is ha

    def test_unknown_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle("nonexistent")

    def test_none_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle(None)


class TestImageEditRoutes:
    @pytest.mark.asyncio
    async def test_edit_reads_upload_before_ray_boundary(self, api):
        import io

        from fastapi import UploadFile

        from modelship.openai.protocol import ImageEditRequest

        handle = MagicMock()
        remote = handle.edit_image.options.return_value.remote
        api.models = {"sdxl": {"sdxl-a1b2c": handle}}
        api._round_robin = {"sdxl": 0}

        request = ImageEditRequest(
            image=UploadFile(file=io.BytesIO(b"IMAGE_BYTES"), filename="i.png"),
            mask=UploadFile(file=io.BytesIO(b"MASK_BYTES"), filename="m.png"),
            prompt="add a hat",
            model="sdxl",
        )
        raw_request = MagicMock()
        raw_request.headers = {}

        with (
            patch("modelship.openai.api.RequestWatcher"),
            patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")) as handle_response,
        ):
            result = await api.create_image_edit(request, raw_request)

        assert result == "OK"
        # The upload bytes must be read in the gateway, not handed to the actor as UploadFile.
        args, _ = remote.call_args
        image_data, mask_data, request_no_file = args[0], args[1], args[2]
        assert image_data == b"IMAGE_BYTES"
        assert mask_data == b"MASK_BYTES"
        # No UploadFile may cross the boundary; the image/mask fields are dropped
        # to None and the bytes are passed separately. (image[] is exclude=True,
        # so it never appears in the dump regardless.)
        dumped = request_no_file.model_dump()
        assert dumped.get("image") is None
        assert dumped.get("mask") is None
        assert "image[]" not in dumped and "image_array" not in dumped
        handle_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_variation_reads_upload_and_omits_mask(self, api):
        import io

        from fastapi import UploadFile

        from modelship.openai.protocol import ImageVariationRequest

        handle = MagicMock()
        remote = handle.vary_image.options.return_value.remote
        api.models = {"sdxl": {"sdxl-a1b2c": handle}}
        api._round_robin = {"sdxl": 0}

        request = ImageVariationRequest(
            image=UploadFile(file=io.BytesIO(b"IMAGE_BYTES"), filename="i.png"),
            model="sdxl",
        )
        raw_request = MagicMock()
        raw_request.headers = {}

        with (
            patch("modelship.openai.api.RequestWatcher"),
            patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")),
        ):
            await api.create_image_variation(request, raw_request)

        args, _ = remote.call_args
        assert args[0] == b"IMAGE_BYTES"


class TestImageFormDecomposition:
    """Exercise the request models through FastAPI's real multipart/form-data
    decomposition (not a direct model_validate), since that is what Open WebUI
    hits and where the `image[]` array field must be picked up."""

    @staticmethod
    def _client():
        import io
        from typing import Annotated

        from fastapi import FastAPI, Form, Request
        from fastapi.testclient import TestClient

        from modelship.openai.protocol import ImageEditRequest, ImageVariationRequest

        app = FastAPI()

        @app.post("/v1/images/edits")
        async def edit(request: Annotated[ImageEditRequest, Form()], raw: Request):
            return {
                "image": request.image.filename if request.image else None,
                # The UploadFile must never survive into model_dump (it would
                # fail to serialize across the Ray process boundary).
                "image_keys_in_dump": [k for k in request.model_dump(exclude={"image", "mask"}) if "image" in k],
            }

        @app.post("/v1/images/variations")
        async def variation(request: Annotated[ImageVariationRequest, Form()], raw: Request):
            return {
                "image": request.image.filename if request.image else None,
                "image_keys_in_dump": [k for k in request.model_dump(exclude={"image"}) if "image" in k],
            }

        return TestClient(app), io

    def test_edit_accepts_bracketed_image_array_field(self):
        # Open WebUI (and OpenAI's gpt-image-1 form) send the upload as `image[]`.
        client, io = self._client()
        resp = client.post(
            "/v1/images/edits",
            data={"prompt": "add a sombrero", "model": "sdxl"},
            files={"image[]": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["image"] == "goat.png"
        assert body["image_keys_in_dump"] == []

    def test_edit_accepts_singular_image_field(self):
        # The legacy DALL·E 2 singular `image` form must keep working.
        client, io = self._client()
        resp = client.post(
            "/v1/images/edits",
            data={"prompt": "add a sombrero", "model": "sdxl"},
            files={"image": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["image"] == "goat.png"

    def test_edit_missing_image_is_422(self):
        client, _ = self._client()
        resp = client.post("/v1/images/edits", data={"prompt": "add a sombrero", "model": "sdxl"})
        assert resp.status_code == 422

    def test_variation_accepts_bracketed_image_array_field(self):
        client, io = self._client()
        resp = client.post(
            "/v1/images/variations",
            data={"model": "sdxl"},
            files={"image[]": ("goat.png", io.BytesIO(b"IMAGE_BYTES"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["image"] == "goat.png"
        assert body["image_keys_in_dump"] == []


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_handle_json_response_directly(self, api):
        from fastapi.responses import JSONResponse

        async def mock_gen():
            yield JSONResponse(content={"data": "test"})

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_embedding_response(self, api):
        from fastapi.responses import JSONResponse

        from modelship.openai.protocol import EmbeddingResponse, UsageInfo

        resp = EmbeddingResponse(
            model="test",
            data=[],
            usage=UsageInfo(prompt_tokens=10, total_tokens=10),
            created=123,
        )

        async def mock_gen():
            yield resp

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        # Check if content matches the model dump
        assert b'"model":"test"' in result.body

    @pytest.mark.asyncio
    async def test_handle_streaming_chat(self, api):
        from fastapi.responses import StreamingResponse

        async def mock_gen():
            yield "data: chunk1\n\n"
            yield "data: chunk2\n\n"
            yield "data: [DONE]\n\n"

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, StreamingResponse)

    @pytest.mark.asyncio
    async def test_raytaskerror_with_value_error_cause_returns_400(self, api):
        import json

        from fastapi.responses import JSONResponse
        from ray.exceptions import RayTaskError

        # Build a RayTaskError whose .cause is a ValueError subclass with a
        # `parameter` attribute — mirrors what VLLMValidationError looks like
        # after Ray transports it across process boundaries.
        class _FakeValidationError(ValueError):
            def __init__(self, message: str, parameter: str) -> None:
                super().__init__(message)
                self.parameter = parameter

        cause = _FakeValidationError("This model's maximum context length is 14512 tokens.", "input_tokens")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def mock_gen():
            if False:
                yield  # pragma: no cover — make this an async generator
            raise err

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "input_tokens"
        assert "maximum context length" in body["error"]["message"]
        watcher.stop.assert_called()

    @pytest.mark.asyncio
    async def test_raytaskerror_with_unknown_cause_returns_500(self, api):
        from fastapi.responses import JSONResponse
        from ray.exceptions import RayTaskError

        cause = RuntimeError("something exploded internally")
        err = RayTaskError(function_name="fn", traceback_str="tb", cause=cause)

        async def mock_gen():
            if False:
                yield  # pragma: no cover
            raise err

        watcher = MagicMock()
        result = await api._handle_response(mock_gen(), watcher, "test-model", "test-endpoint")

        assert isinstance(result, JSONResponse)
        assert result.status_code == 500
