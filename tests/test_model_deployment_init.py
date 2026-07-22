"""ModelDeployment.__init__: a ModelDownloadError must never be
reported to the coordinator as fatal, so it's retried next pass instead of
evicted from the effective config."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.infer.model_deployment import ModelDeployment
from modelship.infer.model_resolver import ModelDownloadError

# Bypass the @serve.deployment wrapper (see test_model_deployment_metrics.py).
_ModelDeployment = ModelDeployment.func_or_class


def _make_config():
    config = MagicMock()
    config.name = "test-model"
    config.loader.value = "vllm"
    return config


def _patch_init_globals(**kwargs):
    # @serve.deployment cloudpickles the class, so the unwrapped __init__ carries
    # a reconstructed globals dict — patching the module attribute wouldn't reach
    # it (same gotcha as test_model_deployment_metrics.py's _patch_gen_metric).
    return patch.dict(_ModelDeployment.__init__.__globals__, kwargs)


@pytest.mark.asyncio
async def test_download_error_does_not_report_fatal():
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ModelDownloadError("network blip"))

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            _spawn_orphan_reaper=MagicMock(return_value=None),
            BaseInfer=mock_base_infer,
            MODEL_LOAD_FAILURES_TOTAL=MagicMock(),
            MODEL_LOAD_DURATION_SECONDS=MagicMock(),
        ),
        patch("modelship.infer.deploy_coordinator.get_or_create_coordinator") as mock_get_coordinator,
        pytest.raises(ModelDownloadError),
    ):
        await _ModelDeployment.__init__(inst, config)

    # The whole point: no coordinator lookup/report happens on this path.
    mock_get_coordinator.assert_not_called()


@pytest.mark.asyncio
async def test_generic_init_failure_reports_fatal():
    """Control case: a non-download init failure is unchanged — still
    reported fatal, still wrapped in RuntimeError."""
    inst = _ModelDeployment.__new__(_ModelDeployment)
    config = _make_config()

    mock_base_infer = MagicMock()
    mock_base_infer.ensure_downloaded = AsyncMock(side_effect=ValueError("bad config"))

    coordinator = MagicMock()
    coordinator.report_fatal_error.remote = AsyncMock()

    with (
        _patch_init_globals(
            configure_logging=MagicMock(),
            stamp_gateway=MagicMock(),
            _spawn_orphan_reaper=MagicMock(return_value=None),
            BaseInfer=mock_base_infer,
            MODEL_LOAD_FAILURES_TOTAL=MagicMock(),
            MODEL_LOAD_DURATION_SECONDS=MagicMock(),
            serve=MagicMock(get_replica_context=MagicMock(return_value=MagicMock(app_name="app"))),
        ),
        patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=coordinator),
        pytest.raises(RuntimeError),
    ):
        await _ModelDeployment.__init__(inst, config)

    coordinator.report_fatal_error.remote.assert_called_once()
