"""Tests for ModelshipAPI model discovery and routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.openai.api import ModelshipAPI

# Access the underlying class, bypassing the @serve.deployment wrapper.
_ModelshipAPI = ModelshipAPI.func_or_class


@pytest.fixture
def api():
    """Create a ModelshipAPI instance with mocked Ray Serve context."""
    with patch("modelship.openai.api.serve.get_replica_context") as mock_ctx:
        mock_ctx.return_value.app_name = "test-gateway"
        return _ModelshipAPI("test-gateway")


class TestAddModels:
    @pytest.mark.asyncio
    async def test_add_single_model(self, api):
        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})

        assert "qwen" in api.models
        assert len(api.models["qwen"]) == 1
        assert api.model_list[0].id == "qwen"

    @pytest.mark.asyncio
    async def test_add_multiple_deployments_same_model(self, api):
        mock_handle_1 = MagicMock()
        mock_handle_2 = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[mock_handle_1, mock_handle_2]):
            await api.add_models({"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"})

        assert len(api.models["qwen"]) == 2
        assert len(api.model_list) == 1

    @pytest.mark.asyncio
    async def test_add_different_models(self, api):
        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"})

        assert "qwen" in api.models
        assert "kokoro" in api.models
        assert len(api.model_list) == 2

    @pytest.mark.asyncio
    async def test_incremental_adds_new_handle_to_existing_model(self, api):
        handle_1 = MagicMock()
        handle_2 = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=handle_1):
            await api.add_models({"qwen-a3f9k": "qwen"})
        with patch("modelship.openai.api.serve.get_app_handle", return_value=handle_2):
            await api.add_models({"qwen-b7x2p": "qwen"})

        assert len(api.models["qwen"]) == 2
        assert api.models["qwen"]["qwen-a3f9k"] is handle_1
        assert api.models["qwen"]["qwen-b7x2p"] is handle_2
        # Only one model card despite two deployments
        assert len(api.model_list) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_skips(self, api):
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=Exception("not found")):
            await api.add_models({"qwen-a3f9k": "qwen"})

        assert "qwen" not in api.models
        assert len(api.model_list) == 0

    @pytest.mark.asyncio
    async def test_records_per_model_load_times_and_ready_timestamp(self, api):
        await api.set_expected_models(["qwen", "kokoro"])
        assert api._expected_set_at is not None
        assert api._all_ready_at is None

        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})
            assert "qwen" in api._model_load_times
            assert api._model_load_times["qwen"] >= 0
            assert api._all_ready_at is None

            await api.add_models({"kokoro-c1m4n": "kokoro"})
            assert "kokoro" in api._model_load_times
            assert api._all_ready_at is not None

    @pytest.mark.asyncio
    async def test_readyz_body_ready_flag(self, api):
        await api.set_expected_models(["qwen"])
        body = api._readyz_body()
        assert body["ready"] is False
        assert body["models_pending"] == ["qwen"]
        assert body["time_to_ready_s"] is None

        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})

        body = api._readyz_body()
        assert body["ready"] is True
        assert body["models_pending"] == []
        assert body["time_to_ready_s"] is not None
        assert "qwen" in body["model_load_times_s"]


class TestRemoveDeployments:
    @pytest.mark.asyncio
    async def test_remove_last_deployment_drops_model(self, api):
        with patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()):
            await api.add_models({"qwen-a3f9k1b2c4": "qwen"})
        assert "qwen" in api.models

        removed = await api.remove_deployments(["qwen-a3f9k1b2c4"])

        assert removed == ["qwen"]
        assert "qwen" not in api.models
        assert api.model_list == []
        assert "qwen" not in api._round_robin

    @pytest.mark.asyncio
    async def test_remove_one_of_many_keeps_model(self, api):
        h1, h2 = MagicMock(), MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[h1, h2]):
            await api.add_models({"qwen-aaaaaaaaaa": "qwen", "qwen-bbbbbbbbbb": "qwen"})

        removed = await api.remove_deployments(["qwen-aaaaaaaaaa"])

        assert removed == []  # model still has a deployment
        assert "qwen" in api.models
        assert list(api.models["qwen"].keys()) == ["qwen-bbbbbbbbbb"]
        assert len(api.model_list) == 1

    @pytest.mark.asyncio
    async def test_remove_unknown_deployment_is_warning(self, api):
        # Should not raise; just logs a warning.
        removed = await api.remove_deployments(["nonexistent-1234567890"])
        assert removed == []

    @pytest.mark.asyncio
    async def test_remove_drops_from_expected_models(self, api):
        await api.set_expected_models(["qwen", "kokoro"])
        with patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()):
            await api.add_models({"qwen-a3f9k1b2c4": "qwen"})

        await api.remove_deployments(["qwen-a3f9k1b2c4"])

        assert api.expected_models == ["kokoro"]


class TestListDeployments:
    @pytest.mark.asyncio
    async def test_returns_app_names_per_model(self, api):
        h1, h2, h3 = MagicMock(), MagicMock(), MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[h1, h2, h3]):
            await api.add_models(
                {
                    "qwen-aaaaaaaaaa": "qwen",
                    "qwen-bbbbbbbbbb": "qwen",
                    "kokoro-cccccccccc": "kokoro",
                }
            )

        listed = await api.list_deployments()

        assert set(listed["qwen"]) == {"qwen-aaaaaaaaaa", "qwen-bbbbbbbbbb"}
        assert listed["kokoro"] == ["kokoro-cccccccccc"]


class TestGatewayReconstruction:
    @pytest.mark.asyncio
    async def test_reconstructs_from_registry_after_restart(self, api):
        # A restarted gateway starts empty; the registry still knows its deployments.
        records = {"qwen-aaaaaaaaaa": "qwen", "embed-bbbbbbbbbb": "embed"}
        with (
            patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=MagicMock()),
            patch("modelship.openai.api.ray.get", return_value=records),
            patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()),
        ):
            api._ensure_reconstructed()

        assert set(api.models) == {"qwen", "embed"}
        assert api._reconstructed is True
        # Recovered set becomes the readiness baseline so the pod reports ready again.
        assert api._readyz_body()["ready"] is True

    @pytest.mark.asyncio
    async def test_retries_when_registry_unavailable(self, api):
        with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", side_effect=RuntimeError):
            api._ensure_reconstructed()
        assert api._reconstructed is False  # not latched — retries on next request
        assert api.models == {}

    @pytest.mark.asyncio
    async def test_skips_reconstruction_when_already_seeded(self, api):
        with patch("modelship.openai.api.serve.get_app_handle", return_value=MagicMock()):
            await api.add_models({"qwen-aaaaaaaaaa": "qwen"})
        with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator") as gc:
            api._ensure_reconstructed()
        gc.assert_not_called()  # driver already seeded us; don't touch the registry


class TestGetHandle:
    @pytest.mark.asyncio
    async def test_round_robin(self, api):
        handle_a = MagicMock()
        handle_b = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[handle_a, handle_b]):
            await api.add_models({"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"})

        assert api._get_handle("qwen") is handle_a
        assert api._get_handle("qwen") is handle_b
        assert api._get_handle("qwen") is handle_a

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
