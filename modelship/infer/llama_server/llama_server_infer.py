import asyncio
import contextlib
import json
import os
import secrets
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import LlamaServerConfig, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.chat_utils import (
    ParsedChatOutput,
    UnsupportedContentError,
    build_from_parsed,
    normalize_chat_messages,
)
from modelship.openai.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    ErrorResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
    create_error_response,
)
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight
from modelship.utils import base_request_id, random_uuid

logger = get_logger("infer.llama_server")

_HEALTH_POLL_INTERVAL_S = 0.5
_STARTUP_TIMEOUT_S = float(os.environ.get("MSHIP_LLAMA_SERVER_STARTUP_TIMEOUT", "900"))
# A child that dies this soon after spawn is presumed to be the free-port
# TOCTOU race (port stolen between our probe close() and its bind()), not a
# real load failure — retry the whole launch with a fresh port.
_EARLY_CRASH_WINDOW_S = 3.0
_LAUNCH_RETRY_LIMIT = 5
_RECENT_LOG_LINES = 50


def _free_port() -> int:
    """Bind an ephemeral port and release it immediately so the child can bind
    it. Racy (TOCTOU) by nature: something else may grab the port before the
    child does. The caller retries the whole launch when that happens."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _EarlyCrashError(RuntimeError):
    """The llama-server child exited within `_EARLY_CRASH_WINDOW_S` of spawn —
    treated as a transient bind race and retried with a fresh port."""


# asyncio only holds a weak reference to a task's coroutine, so a fire-and-forget
# task with no other referent can be GC'd before it runs. Keep a strong reference
# here for the lifetime of each client-close task, started from the (sync) shutdown().
_pending_client_closes: set[asyncio.Task] = set()


class LlamaServerInfer(BaseInfer):
    """Drives a `llama-server` subprocess over its native OpenAI-compatible
    HTTP API. Unlike `llama_cpp` (in-process `Llama` bindings + modelship's
    own tool-call/reasoning parsers), llama-server does its own chat
    templating, tool-call, and reasoning parsing — this loader is a thin,
    concurrency-safe proxy that projects its responses onto modelship's
    protocol models (never relaying its JSON/SSE verbatim, which would leak
    llama.cpp-only extension fields like `timings` to clients)."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        user_config = model_config.llama_server_config or LlamaServerConfig()
        user_overrides = user_config.model_dump(exclude_unset=True)

        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        self.config = user_config.model_copy(update=merged)

        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._api_key: str = secrets.token_hex(32)
        self._client: httpx.AsyncClient | None = None
        self._log_lock = threading.Lock()
        self._recent_log_lines: deque[str] = deque(maxlen=_RECENT_LOG_LINES)
        self._log_threads: list[threading.Thread] = []

    def shutdown(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                logger.info("Shutting down llama-server for %s", self.model_config.name)
                self._proc.terminate()

                proc = self._proc

                def _wait_and_kill():
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(Exception):
                            proc.kill()
                    except Exception:
                        pass

                thread = threading.Thread(target=_wait_and_kill, daemon=True)
                thread.start()

            # Close pipes to immediately unblock and terminate log-draining threads
            if self._proc.stdout is not None:
                with contextlib.suppress(Exception):
                    self._proc.stdout.close()
            if self._proc.stderr is not None:
                with contextlib.suppress(Exception):
                    self._proc.stderr.close()

        self._proc = None
        self._log_threads = []
        if self._client is not None:
            client, self._client = self._client, None
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(client.aclose())
                if _pending_client_closes is not None:
                    _pending_client_closes.add(task)
                    task.add_done_callback(
                        lambda t: _pending_client_closes.discard(t) if _pending_client_closes is not None else None
                    )
            except (RuntimeError, AttributeError, TypeError):
                pass  # no running loop or closed loop (e.g. interpreter teardown) — let GC reclaim the socket

    def __del__(self):
        with contextlib.suppress(BaseException):
            self.shutdown()

    async def start(self) -> None:
        binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
        if not binary or not os.path.isfile(binary):
            raise ValueError(
                f"llama_server deployment '{self.model_config.name}' requires MSHIP_LLAMA_SERVER_BIN "
                f"to point at a llama-server executable; got {binary!r}. See docs/development.md."
            )

        model_path = self.model_config._resolved_path
        if not model_path:
            raise ValueError(
                f"LlamaServer deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        logger.info("Starting llama-server for model: %s", self.model_config.name)
        last_error: Exception | None = None
        try:
            for attempt in range(1, _LAUNCH_RETRY_LIMIT + 1):
                try:
                    await self._launch(binary, model_path)
                    await self._wait_healthy()
                    break
                except _EarlyCrashError as e:
                    last_error = e
                    logger.warning(
                        "llama-server for '%s' exited immediately on attempt %d/%d (likely a port race); retrying: %s",
                        self.model_config.name,
                        attempt,
                        _LAUNCH_RETRY_LIMIT,
                        e,
                    )
                    self.shutdown()
            else:
                raise RuntimeError(
                    f"llama-server for '{self.model_config.name}' failed to start after "
                    f"{_LAUNCH_RETRY_LIMIT} attempts: {last_error}"
                )
        except Exception:
            self.shutdown()
            raise

        assert self._port is not None
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self._port}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(timeout=None, connect=10.0),
        )
        self._set_max_context_length(self.config.n_ctx)

    async def _launch(self, binary: str, model_path: str) -> None:
        loop = asyncio.get_running_loop()
        port = await loop.run_in_executor(None, _free_port)
        self._port = port

        args = [
            binary,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-m",
            model_path,
            "-c",
            str(self.config.n_ctx * self.config.parallel),
            "-b",
            str(self.config.n_batch),
            "--parallel",
            str(self.config.parallel),
            "--jinja",
            "--reasoning-format",
            "auto",
            "--no-webui",
            "--api-key",
            self._api_key,
        ]
        if self.model_config.num_gpus > 0:
            args += ["-ngl", str(self.config.n_gpu_layers)]
        else:
            # Ray only sets CUDA_VISIBLE_DEVICES for actors that reserve GPUs, so
            # a num_gpus=0 deployment may still see every GPU — force no offload.
            args += ["-ngl", "0"]
        if self.config.chat_template:
            flag = "--chat-template-file" if os.path.isfile(self.config.chat_template) else "--chat-template"
            args += [flag, self.config.chat_template]
        if self.config.mmproj:
            from modelship.infer.model_resolver import resolve_model_source

            mmproj_ref = self.config.mmproj
            try:
                mmproj_path = await loop.run_in_executor(None, lambda: resolve_model_source(mmproj_ref))
            except Exception as e:
                logger.warning("Failed to resolve mmproj %r, using as is: %s", mmproj_ref, e)
                mmproj_path = mmproj_ref
            args += ["--mmproj", mmproj_path]
        if self.model_config.usecase == ModelUsecase.embed:
            args += ["--embedding"]
        args += list(self.config.extra_args)

        logger.info("llama-server launch args for '%s': %s", self.model_config.name, _redact(args))
        self._proc = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                errors="replace",
            ),
        )
        with self._log_lock:
            self._recent_log_lines.clear()
        assert self._proc is not None
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        thread_stdout = threading.Thread(
            target=self._drain_stream,
            args=(self._proc.stdout, "stdout"),
            daemon=True,
        )
        thread_stderr = threading.Thread(
            target=self._drain_stream,
            args=(self._proc.stderr, "stderr"),
            daemon=True,
        )
        thread_stdout.start()
        thread_stderr.start()
        self._log_threads = [thread_stdout, thread_stderr]

    def _drain_stream(self, stream: Any, tag: str) -> None:
        """Consume a pipe to TRACE-level logs so the child never blocks on a
        full pipe buffer during chatty load-time output. Runs in a worker
        thread; `stream.readline()` returns '' at EOF (including when the
        child dies and we close the pipe from the main thread)."""
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\n")
                logger.log(TRACE, "[%s:%s] %s", self.model_config.name, tag, line)
                with self._log_lock:
                    self._recent_log_lines.append(line)
        except (ValueError, OSError):
            pass  # stream closed from the main thread while we were reading
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    async def _wait_healthy(self) -> None:
        assert self._proc is not None
        assert self._port is not None
        spawned_at = time.monotonic()
        deadline = spawned_at + _STARTUP_TIMEOUT_S

        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self._port}",
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as probe:
            while True:
                rc = self._proc.poll()
                if rc is not None:
                    with self._log_lock:
                        tail = "\n".join(self._recent_log_lines)
                    message = f"llama-server for '{self.model_config.name}' exited (rc={rc}) during startup: {tail}"
                    if time.monotonic() - spawned_at < _EARLY_CRASH_WINDOW_S:
                        raise _EarlyCrashError(message)
                    raise RuntimeError(message)

                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"llama-server for '{self.model_config.name}' did not become healthy within "
                        f"{_STARTUP_TIMEOUT_S}s"
                    )

                try:
                    resp = await probe.get("/health", timeout=2.0)
                    if resp.status_code == 200:
                        logger.info("llama-server healthy for '%s' on port %d", self.model_config.name, self._port)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

    async def warmup(self) -> None:
        if self.model_config.usecase == ModelUsecase.embed:
            logger.info("Warming up llama-server embedding model: %s", self.model_config.name)
            request = EmbeddingRequest(
                model=self.model_config.name,
                input="warmup",
            )
            result = await self.create_embedding(request, RawRequestProxy(None, {}))
            if isinstance(result, ErrorResponse):
                logger.warning("warmup embedding failed for '%s': %s", self.model_config.name, result.error.message)
            else:
                logger.info("warmup embedding done for %s", self.model_config.name)
            return

        logger.info("Warming up llama-server chat model: %s", self.model_config.name)
        request = ChatCompletionRequest(
            model=self.model_config.name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        result = await self.create_chat_completion(request, RawRequestProxy(None, {}))
        if isinstance(result, ErrorResponse):
            logger.warning("warmup chat failed for '%s': %s", self.model_config.name, result.error.message)
        else:
            logger.info("warmup chat done for %s", self.model_config.name)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | EmbeddingResponse:
        if self._client is None:
            return await super().create_embedding(request, raw_request)

        request_id = f"embd-{base_request_id(raw_request)}"
        logger.info("embedding request %s", request_id)

        payload = request.model_dump(exclude_none=True, exclude={"model"})
        payload["model"] = self.model_config.name

        try:
            resp = await self._client.post("/v1/embeddings", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.warning("embedding request %s failed: %s", request_id, detail)
            return create_error_response(detail, status_code=e.response.status_code)
        except httpx.HTTPError as e:
            logger.warning("embedding request %s failed: %s", request_id, e)
            return create_error_response(f"llama-server request failed: {e}", status_code=502)

        data = resp.json()
        logger.log(TRACE, "embedding response %s: %s", request_id, data)
        if isinstance(data, dict) and "error" in data:
            error_data = data["error"] or {}
            message = error_data.get("message") if isinstance(error_data, dict) else str(error_data)
            logger.warning("embedding request %s failed with inline error: %s", request_id, message)
            return create_error_response(message or "Unknown error returned from llama-server", status_code=502)
        return _project_embedding_response(data, model_name=self.model_config.name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if self._client is None:
            return await super().create_chat_completion(request, raw_request)

        request_id = f"chat-{base_request_id(raw_request)}"
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)
        logger.log(
            TRACE,
            "chat request %s: messages=%s tools=%s tool_choice=%s",
            request_id,
            request.messages,
            request.tools,
            request.tool_choice,
        )

        supports_image = bool(self.config.mmproj)
        try:
            messages = normalize_chat_messages(request.messages, supports_image=supports_image, supports_audio=False)
        except UnsupportedContentError as e:
            logger.warning("chat request %s rejected: %s", request_id, e)
            return create_error_response(e)

        payload = _build_payload(request, messages, model_name=self.model_config.name)

        if request.stream:
            return self._stream_chat_completion(payload, request_id)

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = _extract_error_detail(e.response)
            logger.warning("chat request %s failed: %s", request_id, detail)
            return create_error_response(detail, status_code=e.response.status_code)
        except httpx.HTTPError as e:
            logger.warning("chat request %s failed: %s", request_id, e)
            return create_error_response(f"llama-server request failed: {e}", status_code=502)

        data = resp.json()
        logger.log(TRACE, "chat response %s: %s", request_id, data)
        if isinstance(data, dict) and "error" in data:
            error_data = data["error"] or {}
            message = error_data.get("message") if isinstance(error_data, dict) else str(error_data)
            logger.warning("chat request %s failed with inline error: %s", request_id, message)
            return create_error_response(message or "Unknown error returned from llama-server", status_code=502)
        return _project_chat_response(data, model_name=self.model_config.name, request_id=request_id)

    async def _stream_chat_completion(self, payload: dict[str, Any], request_id: str) -> AsyncGenerator[str, None]:
        assert self._client is not None
        buffered: list[str] = []
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    await e.response.aread()
                    detail = _extract_error_detail(e.response)
                    logger.warning("chat request %s failed: %s", request_id, detail)
                    yield _encode_error(detail)
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("chat request %s: unparseable stream chunk: %r", request_id, data_str)
                        continue
                    if isinstance(data, dict) and "error" in data:
                        error_data = data["error"] or {}
                        message = error_data.get("message") if isinstance(error_data, dict) else str(error_data)
                        logger.warning("chat request %s failed mid-stream with error: %s", request_id, message)
                        yield _encode_error(message or "Unknown mid-stream error from llama-server")
                        yield "data: [DONE]\n\n"
                        return
                    chunk = _project_stream_chunk(data, model_name=self.model_config.name)
                    for choice in chunk.choices:
                        if choice.delta.content:
                            buffered.append(choice.delta.content)
                    yield _encode_chunk(chunk)
                yield "data: [DONE]\n\n"
        except httpx.HTTPError as e:
            logger.warning("chat request %s failed mid-stream: %s", request_id, e)
            yield _encode_error(f"llama-server request failed: {e}")
            yield "data: [DONE]\n\n"
        finally:
            logger.log(TRACE, "chat response %s (stream): %r", request_id, "".join(buffered))


# ---------------------------------------------------------------------------
# Request / response projection — never relay llama-server's JSON verbatim,
# it's `extra="allow"`-shaped and would leak extension fields (e.g. `timings`).
# ---------------------------------------------------------------------------


def _redact(args: list[str]) -> list[str]:
    redacted = list(args)
    for i, arg in enumerate(redacted):
        if arg == "--api-key" and i + 1 < len(redacted):
            redacted[i + 1] = "***"
    return redacted


def _build_payload(request: ChatCompletionRequest, messages: list[dict], *, model_name: str) -> dict[str, Any]:
    # If request.logprobs is True, we want to forward logprobs and top_logprobs (if > 0) to llama-server.
    exclude_fields = {"messages", "model", "logprobs", "top_logprobs"}
    payload = request.model_dump(exclude_none=True, exclude=exclude_fields)
    payload["messages"] = messages
    payload["model"] = model_name

    if request.logprobs:
        payload["logprobs"] = True
        if request.top_logprobs is not None and request.top_logprobs > 0:
            payload["top_logprobs"] = request.top_logprobs

    return payload


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    except Exception:
        pass
    return response.text or f"HTTP {response.status_code}"


def _project_tool_calls(raw_tool_calls: list[dict] | None) -> list[ToolCall]:
    tool_calls = []
    for tc in raw_tool_calls or []:
        function = tc.get("function") or {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id") or f"chatcmpl-tool-{random_uuid()}",
                type="function",
                function=FunctionCall(name=function.get("name", ""), arguments=function.get("arguments", "")),
            )
        )
    return tool_calls


def _project_usage(raw_usage: dict | None) -> UsageInfo:
    usage = raw_usage or {}
    return UsageInfo(
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
    )


def _project_chat_response(data: dict, *, model_name: str, request_id: str) -> ChatCompletionResponse:
    choices = []
    finish_reasons = []
    logprobs_list = []
    for choice in data.get("choices", []):
        message = choice.get("message") or {}
        dto = ParsedChatOutput(
            content=message.get("content"),
            reasoning=message.get("reasoning_content"),
            tool_calls=_project_tool_calls(message.get("tool_calls")),
        )
        choices.append(dto)
        finish_reasons.append(choice.get("finish_reason") or "stop")

        choice_logprobs = None
        if choice.get("logprobs") is not None:
            try:
                choice_logprobs = ChatCompletionLogProbs.model_validate(choice.get("logprobs"))
            except Exception as e:
                logger.warning("Failed to validate choice logprobs: %s", e)
        logprobs_list.append(choice_logprobs)

    return build_from_parsed(
        request_id=data.get("id") or request_id,
        model_name=model_name,
        choices=choices,
        usage=_project_usage(data.get("usage")),
        finish_reasons=finish_reasons,
        created=data.get("created"),
        logprobs=logprobs_list,
    )


def _project_stream_chunk(data: dict, *, model_name: str) -> ChatCompletionStreamResponse:
    choices = []
    for choice in data.get("choices", []):
        delta = choice.get("delta") or {}
        tool_calls = []
        for i, tc in enumerate(delta.get("tool_calls") or []):
            function = tc.get("function") or {}
            tool_calls.append(
                DeltaToolCall(
                    index=tc.get("index", i),
                    id=tc.get("id"),
                    type="function" if tc.get("type", "function") == "function" else None,
                    function=DeltaFunctionCall(name=function.get("name"), arguments=function.get("arguments")),
                )
            )

        choice_logprobs = None
        if choice.get("logprobs") is not None:
            try:
                choice_logprobs = ChatCompletionLogProbs.model_validate(choice.get("logprobs"))
            except Exception as e:
                logger.warning("Failed to validate stream choice logprobs: %s", e)

        choices.append(
            ChatCompletionResponseStreamChoice(
                index=choice.get("index", 0),
                delta=DeltaMessage(
                    role=delta.get("role"),
                    content=delta.get("content"),
                    reasoning=delta.get("reasoning_content"),
                    tool_calls=tool_calls or None,
                ),
                logprobs=choice_logprobs,
                finish_reason=choice.get("finish_reason"),
            )
        )
    return ChatCompletionStreamResponse(
        model=model_name,
        choices=choices,
        usage=_project_usage(data["usage"]) if data.get("usage") else None,
    )


def _project_embedding_response(data: dict, *, model_name: str) -> EmbeddingResponse:
    raw_data_list = data.get("data", [])
    projected_data = []
    for item in raw_data_list:
        projected_data.append(
            EmbeddingResponseData(
                index=item.get("index", 0), embedding=item.get("embedding", []), object=item.get("object", "embedding")
            )
        )
    raw_usage = data.get("usage") or {}
    usage = UsageInfo(
        prompt_tokens=raw_usage.get("prompt_tokens", 0) or 0,
        completion_tokens=raw_usage.get("completion_tokens", 0) or 0,
        total_tokens=raw_usage.get("total_tokens", 0) or 0,
    )
    return EmbeddingResponse(
        id=data.get("id") or f"embd-{random_uuid()}",
        object=data.get("object", "list"),
        created=data.get("created") or int(time.time()),
        model=model_name,
        data=projected_data,
        usage=usage,
    )


def _encode_chunk(chunk: ChatCompletionStreamResponse) -> str:
    return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"


def _encode_error(detail: str) -> str:
    return f"data: {json.dumps({'error': {'message': detail, 'type': 'api_error'}})}\n\n"
