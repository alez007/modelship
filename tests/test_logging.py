"""Tests for the modelship.logging module."""

import json
import logging
import os
import socket
from logging.handlers import SysLogHandler
from unittest.mock import patch

import pytest

from modelship.logging import (
    _LIB_ENV_VARS,
    _LIB_LOGGERS,
    _LOWERCASE_LEVEL_LIBS,
    ModelshipJsonFormatter,
    ModelshipTextFormatter,
    RequestIdFilter,
    _parse_syslog_target,
    _setup_otel,
    configure_logging,
    get_logger,
    propagate_lib_log_env,
    request_id_var,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the modelship logger and _configured flag between tests."""
    import modelship.logging as yl

    yl._configured = False
    root = logging.getLogger("modelship")
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    root.propagate = True
    saved_lib_levels = {name: logging.getLogger(name).level for name in _LIB_LOGGERS}
    saved_env = {k: os.environ.get(k) for k in _LIB_ENV_VARS}
    # Clear so each test exercises a clean setdefault path. Importing mship_deploy.py
    # in another test runs propagate_lib_log_env() and pollutes os.environ for
    # the rest of the pytest session.
    for k in _LIB_ENV_VARS:
        os.environ.pop(k, None)
    token = request_id_var.set(None)
    yield
    request_id_var.reset(token)
    yl._configured = False
    root.handlers.clear()
    for name, lvl in saved_lib_levels.items():
        logging.getLogger(name).setLevel(lvl)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestGetLogger:
    def test_returns_modelship_prefixed_logger(self):
        log = get_logger("api")
        assert log.name == "modelship.api"

    def test_nested_name(self):
        log = get_logger("infer.vllm")
        assert log.name == "modelship.infer.vllm"


class TestConfigureLogging:
    def test_sets_up_handler(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert root.level == logging.INFO
        assert root.propagate is False

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "DEBUG"})
    def test_respects_log_level_env(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert root.level == logging.DEBUG

    def test_lib_loggers_default_to_warning(self):
        configure_logging()
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING
        for env_var, lib_name in _LIB_ENV_VARS.items():
            expected = "warning" if lib_name in _LOWERCASE_LEVEL_LIBS else "WARNING"
            assert os.environ.get(env_var) == expected

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "DEBUG"})
    def test_lib_loggers_info_at_debug(self):
        configure_logging()
        assert logging.getLogger("modelship").level == logging.DEBUG
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.INFO

    @patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "TRACE"})
    def test_trace_mode_sets_trace_app_debug_libs(self):
        from modelship.logging import TRACE

        configure_logging()
        assert logging.getLogger("modelship").level == TRACE
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG
        for env_var, lib_name in _LIB_ENV_VARS.items():
            expected = "debug" if lib_name in _LOWERCASE_LEVEL_LIBS else "DEBUG"
            assert os.environ.get(env_var) == expected

    @patch.dict(os.environ, {"MSHIP_LOG_FORMAT": "json"})
    def test_json_format(self):
        configure_logging()
        root = logging.getLogger("modelship")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, ModelshipJsonFormatter)

    def test_text_format_default(self):
        configure_logging()
        root = logging.getLogger("modelship")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, ModelshipTextFormatter)


class TestVllmConfigureLoggingOptOut:
    """vLLM runs logging.config.dictConfig() at import time. Python's dictConfig
    unconditionally calls _clearExistingHandlers() -> logging.shutdown(), which
    closes *every* live handler — including Ray Serve's MemoryHandler on the
    `ray.serve` logger — and MemoryHandler.close() sets target=None. shutdown()
    does NOT remove the handler from the logger, so Ray's
    get_component_logger_file_path() later dereferences handler.target.baseFilename
    and raises AttributeError, killing the replica's is_allocated() health check.

    This only bites the vLLM loader (llama_cpp etc. never import vllm) and only on
    the head-restart recovery path: on a cold deploy is_allocated() runs before the
    replica __init__ imports vLLM; on recovery the replica re-imports vLLM first, so
    the nulled target is hit. We set VLLM_CONFIGURE_LOGGING=0 in propagate_lib_log_env
    (which runs before any vLLM import, in both driver and replica) so vLLM skips its
    dictConfig and never touches Ray's handler. Log levels are unaffected — modelship
    sets the vllm logger level directly and via VLLM_LOGGING_LEVEL.
    """

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_CONFIGURE_LOGGING", None)
            propagate_lib_log_env("INFO")
            assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"

    def test_explicit_user_value_wins(self):
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            propagate_lib_log_env("INFO")
            assert os.environ["VLLM_CONFIGURE_LOGGING"] == "1"


class TestRequestIdFilter:
    def test_injects_request_id(self):
        filt = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        request_id_var.set("abc-123")
        filt.filter(record)
        assert record.request_id == "abc-123"

    def test_none_when_not_set(self):
        filt = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        filt.filter(record)
        assert record.request_id is None


class TestModelshipJsonFormatter:
    def test_produces_valid_json(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello %s", ("world",), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "modelship.api"
        assert parsed["message"] == "hello world"
        assert "pid" in parsed

    def test_includes_request_id(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = "req-456"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "req-456"

    def test_excludes_request_id_when_none(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "request_id" not in parsed

    def test_includes_exception(self):
        formatter = ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("modelship.api", logging.ERROR, "", 0, "error", (), sys.exc_info())
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]


class TestModelshipTextFormatter:
    def test_basic_format(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "INFO" in output
        assert "modelship.api" in output
        assert "hello" in output

    def test_includes_request_id(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = "req-789"
        output = formatter.format(record)
        assert "[req-789]" in output

    def test_excludes_request_id_when_none(self):
        formatter = ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("modelship.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "None" not in output


class TestEndToEnd:
    def test_log_output_with_request_id(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("e2e-test-id")
        log.info("end to end")
        captured = capsys.readouterr()
        assert "e2e-test-id" in captured.err
        assert "end to end" in captured.err

    @patch.dict(os.environ, {"MSHIP_LOG_FORMAT": "json"})
    def test_json_log_output(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("json-test-id")
        log.info("json test")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["request_id"] == "json-test-id"
        assert parsed["message"] == "json test"


class TestParseSyslogTarget:
    def test_udp_default(self):
        handler = _parse_syslog_target("syslog://192.168.1.50:514")
        assert handler.address == ("192.168.1.50", 514)
        assert handler.socktype == socket.SOCK_DGRAM

    @patch("modelship.logging.SysLogHandler.createSocket")
    def test_tcp(self, _mock_create):
        handler = _parse_syslog_target("syslog+tcp://127.0.0.1:1514")
        assert handler.address == ("127.0.0.1", 1514)
        assert handler.socktype == socket.SOCK_STREAM

    def test_default_port(self):
        handler = _parse_syslog_target("syslog://127.0.0.1")
        assert handler.address == ("127.0.0.1", 514)
        assert handler.socktype == socket.SOCK_DGRAM

    def test_default_host(self):
        handler = _parse_syslog_target("syslog://")
        assert handler.address == ("localhost", 514)


class TestSyslogLogTarget:
    @patch.dict(os.environ, {"MSHIP_LOG_TARGET": "syslog://127.0.0.1:5140"})
    def test_configure_creates_syslog_handler(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], SysLogHandler)

    def test_console_default(self):
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert not isinstance(root.handlers[0], SysLogHandler)


class TestOtelSetup:
    def test_warns_when_packages_missing(self, capsys):
        """When otel packages aren't installed, _setup_otel logs a warning and adds no handler."""
        root = logging.getLogger("modelship")
        root.handlers.clear()
        sh = logging.StreamHandler()
        root.addHandler(sh)
        root.setLevel(logging.DEBUG)

        _setup_otel(root, "http://localhost:4317", logging.INFO)

        captured = capsys.readouterr()
        assert "opentelemetry packages are not installed" in captured.err
        assert len(root.handlers) == 1

    @patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"})
    @patch("modelship.logging._setup_otel")
    def test_configure_calls_setup_otel(self, mock_setup):
        configure_logging()
        mock_setup.assert_called_once_with(
            logging.getLogger("modelship"),
            "http://collector:4317",
            logging.INFO,
        )

    @patch.dict(os.environ, {}, clear=False)
    def test_configure_skips_otel_when_not_set(self):
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        configure_logging()
        root = logging.getLogger("modelship")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
