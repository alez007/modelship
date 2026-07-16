import base64
import concurrent.futures
import io
import json
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml

from openai import OpenAI

OPENAI_API_BASE = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"
# Ray Serve's REST status API, served by the head dashboard on 8265 (the
# mship_cluster fixture starts the head with the dashboard explicitly enabled).
# Returns ServeInstanceDetails — per-application/-deployment replica states —
# which the autoscaling test polls to observe scale out/in.
SERVE_STATUS_URL = "http://localhost:8265/api/serve/applications/"

# Per-model configs. Each `Deployer.deploy(*names)` call writes one of these
# (or a subset) into a one-shot models.yaml and runs `mship_deploy.py
# --reconcile` to swap the currently-deployed set in-place.
MODEL_CONFIGS: dict[str, dict] = {
    "chat-capable": {
        "name": "chat-capable",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        # num_gpus is also wired into vllm's gpu_memory_utilization; 0.1 leaves no room for KV cache
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 2048,
            "enforce_eager": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    },
    "chat-reasoning": {
        "name": "chat-reasoning",
        # Qwen3-0.6B is the smallest reasoning-capable model in the Qwen3
        # family; it natively emits `<think>...</think>`. Reasoning chains
        # need headroom so `max_model_len` is bumped.
        "model": "Qwen/Qwen3-0.6B",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 4096,
            "enforce_eager": True,
            "enable_reasoning": True,
            "reasoning_parser": "deepseek_r1",
        },
    },
    "chat-vlm": {
        "name": "chat-vlm",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 1,
        "vllm_engine_kwargs": {
            "max_model_len": 8192,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": 2},
            "mm_processor_kwargs": {"min_pixels": 50176, "max_pixels": 200704},
        },
    },
    "autoscale-llama": {
        "name": "autoscale-llama",
        # Tiny CPU GGUF so the host can hold several replicas (1 cpu each, up to
        # max_replicas) at once. autoscaling_config replaces num_replicas:
        # target_ongoing_requests=1 makes a handful of concurrent requests
        # exceed one replica's setpoint and drive scale-out; the short delays
        # keep the test's poll windows tractable.
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 1,
            "upscale_delay_s": 2,
            "downscale_delay_s": 10,
        },
    },
    "chat-llama-server": {
        "name": "chat-llama-server",
        # Qwen3-0.6B GGUF through the llama_server loader: a llama-server
        # subprocess doing its own chat templating, tool-call, and reasoning
        # parsing (`--jinja --reasoning-format auto`). `parallel: 4` exercises
        # the loader's headline capability: true multi-slot concurrency
        # instead of a single asyncio.Lock serializing every request. n_ctx is per-slot (the
        # loader launches with `-c n_ctx*parallel`), bumped for reasoning
        # headroom.
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 2,
        "llama_server_config": {
            "n_ctx": 4096,
            "parallel": 4,
        },
    },
    "chat-llama-server-plain": {
        "name": "chat-llama-server-plain",
        # Same non-reasoning Qwen2.5-0.5B GGUF as chat-llama-server, through
        # the llama_server loader. Used for the response_format tests, which
        # need a model that doesn't emit a `<think>...</think>` preamble.
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "chat-llama-server-gpu": {
        "name": "chat-llama-server-gpu",
        # Same GGUF as chat-llama-server-plain, on a whole GPU — exercises the
        # llama_server loader's offload path (actor GPU allocation, -ngl
        # honored).
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_gpus": 1,
        "num_cpus": 1,
    },
    "embed-model-llama-server": {
        "name": "embed-model-llama-server",
        # Real embeddings through a live llama-server subprocess (`--embedding`)
        # — the existing `test_embeddings` integration test only exercises the
        # vllm loader; llama_server's B4 embeddings support was otherwise
        # only unit-tested against a mocked httpx transport.
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf",
        "usecase": "embed",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "embed-model": {
        "name": "embed-model",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "usecase": "embed",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "stt-model": {
        "name": "stt-model",
        "model": "openai/whisper-tiny",
        "usecase": "transcription",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "tts-model": {
        "name": "tts-model",
        "model": "hexgrad/Kokoro-82M",
        "usecase": "tts",
        "loader": "custom",
        "plugin": "kokoroonnx",
        "num_cpus": 1,
        "plugin_config": {"onnx_provider": "CPUExecutionProvider"},
    },
    "image-model": {
        "name": "image-model",
        "model": "stabilityai/sdxl-turbo",
        "usecase": "image",
        "loader": "diffusers",
        "num_gpus": 1,
        "diffusers_config": {"num_inference_steps": 2, "guidance_scale": 0.0},
    },
    "image-cpu-model": {
        "name": "image-cpu-model",
        # SD2.1 packaged as a single-file sd.cpp GGUF (CLIP + UNet + VAE bundled).
        # CPU-only; few steps + small size keep the integration run tractable.
        "model": "jiaowobaba02/stable-diffusion-v2-1-GGUF:*q4_1.gguf",
        "usecase": "image",
        "loader": "stable_diffusion_cpp",
        "num_cpus": 4,
        "stable_diffusion_cpp_config": {"sample_steps": 6, "cfg_scale": 7.0},
    },
}


class _Deployer:
    """Owns the per-test reconcile cycle: writes a one-shot models.yaml with
    the requested set and runs `mship_deploy.py --reconcile` synchronously
    against the already-running gateway. Re-deploying the same set is a no-op."""

    def __init__(self, tmp_dir: Path) -> None:
        self._tmp = tmp_dir
        self._current: frozenset[str] = frozenset()

    def deploy(self, *model_names: str) -> None:
        wanted = frozenset(model_names)
        if wanted == self._current:
            return

        slug = "+".join(sorted(wanted)) or "empty"
        config_path = self._tmp / f"models-{slug}.yaml"
        log_path = self._tmp / f"reconcile-{slug}.log"
        with open(config_path, "w") as f:
            yaml.dump({"models": [MODEL_CONFIGS[n] for n in sorted(wanted)]}, f)

        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "mship_deploy.py",
                    "--config",
                    str(config_path),
                    "--reconcile",
                    "--replace-strategy",
                    "stop_start",
                    "--prune-ray-sessions",
                    "false",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        if result.returncode != 0:
            tail = log_path.read_text()[-4000:]
            pytest.fail(
                f"mship_deploy --reconcile failed for {slug} (exit {result.returncode}).\n"
                f"Log file: {log_path}\nLast 4KB:\n{tail}"
            )
        self._current = wanted


@pytest.fixture(scope="session")
def mship_cluster(tmp_path_factory):
    """Start a Ray cluster and a long-lived `mship_deploy` operator process
    bound to an empty models.yaml — owns the gateway via the fresh-install
    path and `signal.pause()`s for the rest of the session. Per-test code
    deploys models additively via `_Deployer.deploy(...)`."""
    tmp_dir = tmp_path_factory.mktemp("mship_integration")
    empty_config = tmp_dir / "empty-models.yaml"
    log_path = tmp_dir / "mship_deploy.log"

    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--disable-usage-stats"], check=True)

    with open(empty_config, "w") as f:
        yaml.dump({"models": []}, f)

    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime, closed in cleanup
    proc = subprocess.Popen(
        # 2 gateway replicas for the whole session so TestGatewayReplicaConsistency
        # can verify the coordinator watch loop converges every replica (and all
        # other tests get free multi-replica coverage). Gateway replicas are
        # num_cpus=0, so this is cheap.
        [
            "uv",
            "run",
            "mship_deploy.py",
            "--config",
            str(empty_config),
            "--gateway-replicas",
            "2",
            "--prune-ray-sessions",
            "false",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def cleanup():
        log_file.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        subprocess.run(["ray", "stop", "--force"], check=False)

    try:
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(HEALTH_URL).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
            cleanup()
            pytest.fail(f"Gateway failed to become ready within timeout.\nLog file: {log_path}\nLast 4KB:\n{tail}")

        yield tmp_dir
    finally:
        cleanup()


@pytest.fixture(scope="session")
def model_deployer(mship_cluster) -> _Deployer:
    return _Deployer(mship_cluster)


@pytest.fixture(scope="session")
def client(mship_cluster) -> OpenAI:
    return OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed")


def _collect_streaming_tool_call(stream) -> dict:
    """Drain an OpenAI streaming response and rebuild the assistant message.

    Returns a dict with: ``content`` (concatenated content deltas),
    ``tool_calls`` (per-index dict of ``{id, name, arguments}`` — arguments
    concatenated across all fragments), ``finish_reason``, ``name_deltas``
    and ``args_deltas`` (counts, used to assert that streaming was actually
    incremental rather than a single buffered emission).
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    name_deltas = 0
    args_deltas = 0
    chunks_with_tool_calls = 0

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            chunks_with_tool_calls += 1
            for tc in delta.tool_calls:
                slot = tool_calls.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id is not None:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                    name_deltas += 1
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
                    args_deltas += 1
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "name_deltas": name_deltas,
        "args_deltas": args_deltas,
        "chunks_with_tool_calls": chunks_with_tool_calls,
    }


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.integration
@pytest.mark.vllm
class TestChatCapable:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_list_models(self, client):
        # With per-model deploys we only assert the currently-deployed model
        # appears in /v1/models, not the full original 7-model set.
        model_ids = [m.id for m in client.models.list().data]
        assert "chat-capable" in model_ids

    def test_chat_completion(self, client):
        completion = client.chat.completions.create(
            model="chat-capable", messages=[{"role": "user", "content": "Hello!"}], max_tokens=10
        )
        assert completion.choices[0].message.content
        assert completion.model == "chat-capable"

    def test_chat_streaming(self, client):
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Tell me a short story."}],
            max_tokens=20,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 0

    def test_tool_calling_success(self, client):
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
        )
        assert completion.choices[0].message.tool_calls
        assert completion.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_tool_calling_streaming_vllm_loader(self, client):
        """Smoke-test that vLLM streaming + tool calling still works through
        the gateway. vLLM emits its own per-token deltas; we only verify that
        the gateway forwards them and that the final assistant message rebuilds
        correctly.
        """
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            stream=True,
        )

        collected = _collect_streaming_tool_call(stream)

        assert collected["tool_calls"], "vLLM should have streamed at least one tool call"
        call_0 = collected["tool_calls"][0]
        assert call_0["name"] == "get_weather"
        parsed_args = json.loads(call_0["arguments"])
        assert "Paris" in parsed_args.get("city", "")
        assert collected["finish_reason"] == "tool_calls"

    def test_response_format_json_object_constrains_unprompted_output(self, client):
        """Prompt asks a natural-language question with no JSON hint. A
        passing test means the grammar constraint — not the prompt — produced
        a JSON object instead of prose.
        """
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_response_format_json_schema_constrains_unprompted_output(self, client):
        """Same intent for json_schema: prompt is natural-language, success
        proves the schema constraint is doing the work.
        """
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}
        assert isinstance(parsed["city"], str) and parsed["city"]
        assert isinstance(parsed["country"], str) and parsed["country"]

    def test_response_format_json_schema_streaming_constrains_unprompted_output(self, client):
        """Same as above but over the streaming path."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            max_tokens=64,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        content = "".join(chunks)
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"answer"}
        assert isinstance(parsed["answer"], str) and parsed["answer"]

    def test_response_format_coexists_with_tool_choice_none(self, client):
        """OpenAI allows response_format alongside tool definitions; with
        tool_choice="none" the model must produce schema-constrained text
        instead of calling the tool. vLLM honors both natively.
        """
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        assert not completion.choices[0].message.tool_calls
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}

    def test_n_greater_than_one_returns_independent_choices(self, client):
        """n>1 needs its own parser instance per choice (see
        engine_ops.make_parsers) — a shared instance would corrupt state
        across choices. Sampling is non-greedy by default so choices should
        actually differ; this also guards against a regression that
        silently collapses every choice onto the same output.
        """
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say one random word."}],
            max_tokens=10,
            n=3,
        )
        assert len(completion.choices) == 3
        assert [c.index for c in completion.choices] == [0, 1, 2]
        assert all(c.message.content for c in completion.choices)

    def test_logprobs_returns_choice_logprobs(self, client):
        """logprobs must be built from the engine's own RequestOutput.logprobs
        (engine_ops.build_chat_logprobs), not silently dropped by the
        non-stream rewire."""
        completion = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=3,
        )
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None and logprobs.content
        first = logprobs.content[0]
        assert isinstance(first.token, str) and first.token
        assert isinstance(first.logprob, float)
        assert 0 < len(first.top_logprobs) <= 3
        assert all(isinstance(tl.token, str) and isinstance(tl.logprob, float) for tl in first.top_logprobs)

    def test_streaming_n_greater_than_one_returns_independent_choices(self, client):
        """Streaming counterpart of the n>1 test above — each choice needs its
        own `Parser` instance (`engine_ops.stream_chat_completion`'s per-choice
        `make_parsers` call), or every choice after the first corrupts onto a
        shared stream state."""
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say one random word."}],
            max_tokens=10,
            n=3,
            stream=True,
        )
        content_by_index: dict[int, str] = {0: "", 1: "", 2: ""}
        finish_reasons: dict[int, str | None] = {}
        for chunk in stream:
            for choice in chunk.choices:
                if choice.delta.content:
                    content_by_index[choice.index] += choice.delta.content
                if choice.finish_reason:
                    finish_reasons[choice.index] = choice.finish_reason
        assert set(finish_reasons) == {0, 1, 2}
        assert all(content_by_index[i] for i in range(3))

    def test_streaming_logprobs_returns_choice_logprobs(self, client):
        """Streaming counterpart of the logprobs test above — logprobs must be
        built per-delta from `RequestOutput.logprobs`, not just on the final
        non-streamed response."""
        stream = client.chat.completions.create(
            model="chat-capable",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=3,
            stream=True,
        )
        seen_logprobs = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].logprobs and chunk.choices[0].logprobs.content:
                seen_logprobs.extend(chunk.choices[0].logprobs.content)
        assert seen_logprobs
        first = seen_logprobs[0]
        assert isinstance(first.token, str) and first.token
        assert isinstance(first.logprob, float)
        assert 0 < len(first.top_logprobs) <= 3


@pytest.mark.integration
@pytest.mark.vllm
class TestChatReasoning:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-reasoning")

    def test_reasoning_completion(self):
        """Non-streaming: vLLM's deepseek_r1 reasoning parser routes the
        `<think>...</think>` block to `message.reasoning` and leaves the
        final answer in `message.content`. Verifies end-to-end wiring of
        `reasoning_parser` through the modelship gateway.
        """
        # Use httpx so we read the raw `reasoning` field — the OpenAI Python
        # SDK doesn't always surface it as a typed attribute.
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 512,
            },
            timeout=120,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning"), f"expected reasoning content, got {message!r}"
        # `<think>` markers must be stripped from both fields.
        assert "<think>" not in (message.get("content") or "")
        assert "<think>" not in message["reasoning"]

    def test_reasoning_streaming(self):
        """Streaming: at least one delta carries `reasoning` and the
        concatenated reasoning text is non-empty when the stream ends.
        Markers must not leak into either field.
        """
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-reasoning",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 512,
                "stream": True,
            },
            timeout=120,
        ) as response:
            assert response.status_code == 200
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            reasoning_deltas = 0
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                    reasoning_deltas += 1
                if delta.get("content"):
                    content_parts.append(delta["content"])

        assert reasoning_deltas >= 1, "expected at least one reasoning delta"
        assert "".join(reasoning_parts).strip(), "expected non-empty reasoning content"
        # Reasoning markers must not leak into either stream.
        assert "<think>" not in "".join(reasoning_parts)
        assert "<think>" not in "".join(content_parts)


# 1x1 red pixel PNG
_RED_PIXEL_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)


@pytest.mark.integration
@pytest.mark.vllm
class TestChatVlm:
    """End-to-end vision: a real Qwen2.5-VL-3B deployment receiving an
    ``image_url`` content part through the modelship gateway."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-vlm")

    def test_chat_with_image_url_returns_response(self, client):
        completion = client.chat.completions.create(
            model="chat-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": _RED_PIXEL_DATA_URI}},
                    ],
                }
            ],
            max_tokens=16,
        )
        assert completion.choices[0].message.content
        assert completion.choices[0].finish_reason in {"stop", "length"}
        assert completion.model == "chat-vlm"

    def test_text_only_request_still_works_on_vlm(self, client):
        completion = client.chat.completions.create(
            model="chat-vlm",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=8,
        )
        assert completion.choices[0].message.content

    def test_image_url_streaming(self, client):
        """Streaming + image input through the gateway end-to-end."""
        stream = client.chat.completions.create(
            model="chat-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": _RED_PIXEL_DATA_URI}},
                    ],
                }
            ],
            max_tokens=16,
            stream=True,
        )
        chunks = [c.choices[0].delta.content for c in stream if c.choices[0].delta.content]
        assert len(chunks) > 0


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServer:
    """End-to-end chat, tool calling, reasoning, and concurrency through the
    `llama_server` loader (a `llama-server` subprocess proxied over its
    native OpenAI-compatible HTTP API). Reasoning and tool-call parsing is
    llama-server's own (`--jinja --reasoning-format auto`).
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server")

    def test_chat_completion(self, client):
        # This deployment is Qwen3-0.6B (reasoning-capable), unlike
        # `chat-llama-server-plain`'s plain Qwen2.5 — it always emits a
        # `<think>...` preamble before content, so the token budget needs
        # headroom for reasoning to finish, not just the answer itself.
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=256,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_loader(self, client):
        """Round-trip a tool call through llama-server's own hermes-style
        parser (`--jinja`, auto-detected from the GGUF's chat template)."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_calling_streaming_llama_server_loader(self, client):
        """Stream a tool call through llama-server and verify the delta
        sequence matches the OpenAI streaming contract, same shape as the
        vLLM loader streaming tests."""
        stream = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
            stream=True,
        )

        collected = _collect_streaming_tool_call(stream)

        assert collected["tool_calls"], (
            f"expected at least one streamed tool call; got content={collected['content']!r}"
        )
        call_0 = collected["tool_calls"][0]
        assert call_0["name"] == "get_weather"
        assert collected["args_deltas"] >= 1
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"

    def test_reasoning_completion_llama_server(self):
        """Non-streaming: llama-server's own `--reasoning-format auto` routes
        the `<think>...</think>` block to `message.reasoning`, with the final
        answer in `message.content` and no marker leakage into either."""
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message.get("reasoning"), f"expected reasoning content, got {message!r}"
        assert "<think>" not in (message.get("content") or "")
        assert "</think>" not in (message.get("content") or "")
        assert "<think>" not in message["reasoning"]
        assert "</think>" not in message["reasoning"]

    def test_reasoning_streaming_llama_server(self):
        """Streaming: at least one delta carries `reasoning`; concatenated
        reasoning is non-empty; markers never leak into either field."""
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-server",
                "messages": [{"role": "user", "content": "Briefly: what is 7 times 8?"}],
                "max_tokens": 1024,
                "stream": True,
            },
            timeout=300,
        ) as response:
            assert response.status_code == 200
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            reasoning_deltas = 0
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("reasoning"):
                    reasoning_parts.append(delta["reasoning"])
                    reasoning_deltas += 1
                if delta.get("content"):
                    content_parts.append(delta["content"])

        assert reasoning_deltas >= 1, "expected at least one reasoning delta"
        assert "".join(reasoning_parts).strip(), "expected non-empty reasoning content"
        assert "<think>" not in "".join(reasoning_parts)
        assert "</think>" not in "".join(reasoning_parts)
        assert "<think>" not in "".join(content_parts)
        assert "</think>" not in "".join(content_parts)

    def test_reasoning_with_tools_llama_server(self, client):
        """Reasoning + tool calling in one round-trip: llama-server populates
        both `message.reasoning` and `message.tool_calls`, with
        `finish_reason="tool_calls"`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        # The OpenAI Python SDK exposes unknown fields via `model_extra`.
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert "<think>" not in reasoning
        tool_calls = message.tool_calls
        assert tool_calls, f"expected a tool call, got content={message.content!r}, reasoning={reasoning!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_markers_inside_reasoning_not_double_counted_llama_server(self, client):
        """Verifies llama-server's own parser doesn't double-count a
        `<tool_call>...</tool_call>` illustration quoted inside `<think>`
        reasoning as a second, real call — a bug pattern plausible for any
        single-pass parser.

        Coaxes the model into illustrating tool-call syntax inside its
        reasoning before making one actual call, and asserts exactly one real
        `tool_calls` entry comes out. Real models are non-deterministic; if
        the prompt fails to produce literal markers in reasoning, the
        marker-routing assertion is skipped rather than flaking — the
        single-tool-call assertion still has value either way.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant with access to tools. When you think inside "
                        "<think>...</think>, FIRST quote one example of tool-call syntax "
                        "verbatim inside angle brackets — e.g. write the literal text "
                        '<tool_call>{"name":"example","arguments":{}}</tool_call> as part '
                        "of your reasoning to remind yourself of the format. THEN decide "
                        "which real tool to call."
                    ),
                },
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning") or ""
        tool_calls = message.tool_calls or []

        assert tool_calls, (
            f"expected exactly one real tool call, got content={message.content!r}, reasoning={reasoning!r}"
        )
        assert len(tool_calls) == 1, (
            f"expected exactly one tool call (markers inside <think> must not be double-counted); "
            f"got {len(tool_calls)} calls={[tc.function.name for tc in tool_calls]}"
        )
        assert tool_calls[0].function.name == "get_weather"
        assert completion.choices[0].finish_reason == "tool_calls"

        if "<tool_call>" in reasoning:
            assert "</tool_call>" in reasoning, (
                f"reasoning has an unmatched <tool_call> marker (open without close): {reasoning!r}"
            )

    def test_response_format_with_reasoning_llama_server(self, client):
        """llama-server handles a JSON-schema `response_format` combined with
        reasoning natively, routing `<think>...</think>` to `message.reasoning`
        and the schema-conforming JSON to `message.content`."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "strict": True,
                },
            },
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert message.content
        parsed = json.loads(message.content)
        assert "answer" in parsed

    def test_tool_choice_required_is_a_noop_on_hermes_family(self, client):
        """Documents a real gap (A1 spike finding): `tool_choice: required` is
        grammar-enforced on harmony-style chat templates (e.g. gpt-oss) but a
        silent no-op on hermes-style ones — Qwen3 (this deployment) included.
        Real grammar forcing makes the free-text branch structurally
        unreachable (`message.content` would be empty/None); on this
        hermes-style model it stays reachable even under `required`, proving
        no grammar constraint was applied. (Verified against a live run: this
        0.6B model is unstable enough to *also* emit a spurious tool call
        alongside genuine free text on an irrelevant prompt — which is a
        model-quality quirk, not evidence of forcing, so this asserts on
        content reachability rather than tool_calls absence.) If llama.cpp
        starts enforcing this for hermes models, `content` goes empty and
        this test fails — update the docs/CLAUDE.md gap notes.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=256,
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_named_function_tool_choice_falls_back_to_auto(self, client):
        """Documents a real gap (A1 spike finding): object-form `tool_choice`
        (named-function forcing) is globally unsupported in llama.cpp b9859 —
        it silently falls back to `auto` rather than forcing the named
        function or erroring. Same content-reachability technique and
        irrelevant-tool prompt as the `required` gap test above — real
        forcing would make the free-text branch structurally unreachable."""
        completion = client.chat.completions.create(
            model="chat-llama-server",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=256,
        )
        message = completion.choices[0].message
        assert message.content, f"expected the free-text branch to stay reachable, got message={message!r}"

    def test_concurrent_requests_are_not_serialized(self, client):
        """The loader's headline capability: llama-server's `--parallel` slots
        let several requests run concurrently instead of being serialized
        behind a single lock. Time one request, then several at once, and
        assert the concurrent batch finishes well under what full
        serialization would take.
        """
        prompt = {
            "model": "chat-llama-server",
            "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
            "max_tokens": 200,
        }

        start = time.monotonic()
        client.chat.completions.create(**prompt)
        baseline = time.monotonic() - start

        concurrency = 3
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(client.chat.completions.create, **prompt) for _ in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        concurrent_elapsed = time.monotonic() - start

        # Full serialization (a single-lock loader's behavior) would take
        # roughly concurrency * baseline; llama-server's parallel slots should
        # keep this well under that.
        assert concurrent_elapsed < baseline * (concurrency - 0.5), (
            f"expected concurrent requests to overlap via llama-server's parallel slots "
            f"(baseline={baseline:.1f}s, {concurrency} concurrent took {concurrent_elapsed:.1f}s)"
        )


@pytest.mark.integration
@pytest.mark.llama_server
class TestResponsesLlamaServer:
    """The /v1/responses adapter is loader-agnostic: same smoke test shape
    as vLLM's, run over the llama_server loader."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server")

    def test_basic_response_through_llama_server(self, client):
        # `chat-llama-server` is Qwen3-0.6B (reasoning-capable) — it always
        # emits a `<think>...` preamble first, so the token budget needs
        # headroom for reasoning to finish, not just the one-word answer.
        resp = client.responses.create(
            model="chat-llama-server",
            input="Say hello in one word.",
            max_output_tokens=512,
        )
        assert resp.status in {"completed", "incomplete"}
        assert resp.output_text.strip()

    def test_streaming_response_through_llama_server(self, client):
        stream = client.responses.create(
            model="chat-llama-server",
            input="Say hello in one word.",
            max_output_tokens=512,
            stream=True,
        )
        text_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.output_text.delta":
                text_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert "".join(text_deltas).strip()
        assert completed is not None
        assert completed.status in {"completed", "incomplete"}


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerResponseFormat:
    """response_format tests for the llama_server loader. Uses
    `chat-llama-server-plain` (non-reasoning Qwen2.5-0.5B) rather than
    `chat-llama-server` (Qwen3, always emits a `<think>...` preamble) —
    response_format + reasoning together is covered separately by
    `TestChatLlamaServer.test_response_format_with_reasoning_llama_server`.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-plain")

    def test_response_format_json_object_without_schema_is_unconstrained(self, client):
        """llama-server's own docs claim bare `{"type": "json_object"}` (no `schema` key) produces
        "plain JSON output" like other OpenAI-inspired providers, but verified
        directly against the b9859 binary (`curl` straight to `/v1/chat/completions`,
        bypassing modelship) this isn't enforced — the model answers in free
        text with no error. Constraining does work once a `schema` key is
        attached to the `response_format` object (an llama-server extension,
        not in the OpenAI spec — `type: json_schema`, which modelship's
        protocol sends for schema-constrained requests, does carry a schema
        and IS honored — see `test_response_format_json_schema_constrains_unprompted_output`).
        If this test starts failing (content parses as JSON), llama-server
        started honoring plain `json_object` — update this note and
        CLAUDE.md/AGENTS.md's llama_server gap list.
        """
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)

    def test_response_format_json_schema_constrains_unprompted_output(self, client):
        """A natural-language question + json_schema → schema-conformant output."""
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}
        assert isinstance(parsed["city"], str) and parsed["city"]
        assert isinstance(parsed["country"], str) and parsed["country"]

    def test_response_format_json_schema_streaming_constrains_unprompted_output(self, client):
        """Same intent on the streaming path."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        stream = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            max_tokens=64,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        content = "".join(chunks)
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"answer"}
        assert isinstance(parsed["answer"], str) and parsed["answer"]

    def test_response_format_coexists_with_tool_choice_none(self, client):
        """tool_choice='none' is the safe escape valve: tools listed but inert,
        schema enforced on content output.
        """
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model="chat-llama-server-plain",
            messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="none",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "location", "schema": schema, "strict": True},
            },
            max_tokens=64,
        )
        assert not completion.choices[0].message.tool_calls
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert set(parsed.keys()) == {"city", "country"}


@pytest.mark.integration
@pytest.mark.llama_server
class TestChatLlamaServerGpu:
    """End-to-end GPU offload through the llama_server loader.

    Same GGUF and tool-calling shape as `TestChatLlamaServerResponseFormat`
    (CPU), but deployed with `num_gpus=1` so the actor gets a whole GPU and
    the loader passes `-ngl` for real offload instead of the forced `-ngl 0`
    it uses when `num_gpus` is `0`.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-server-gpu")

    def test_chat_completion(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
        )
        content = completion.choices[0].message.content
        assert content
        assert "Paris" in content

    def test_tool_calling_llama_server_gpu_loader(self, client):
        completion = client.chat.completions.create(
            model="chat-llama-server-gpu",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.integration
@pytest.mark.llama_server
def test_embeddings_llama_server(client, model_deployer):
    """Real embeddings through a live llama-server subprocess (`--embedding`).
    `test_embeddings` only exercises the vllm loader; this is the
    first live-binary coverage of llama_server's B4 embeddings support
    (previously only unit-tested against a mocked httpx transport)."""
    model_deployer.deploy("embed-model-llama-server")
    response = client.embeddings.create(model="embed-model-llama-server", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


@pytest.mark.integration
def test_embeddings(client, model_deployer):
    model_deployer.deploy("embed-model")
    response = client.embeddings.create(model="embed-model", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


@pytest.mark.integration
class TestAudio:
    """tts-model and stt-model are both small and fit comfortably side-by-side,
    so they share a class — `test_audio_transcription` re-uses the TTS output
    as its input."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-model")

    def test_audio_speech(self, client):
        response = client.audio.speech.create(model="tts-model", voice="af_bella", input="Hello from integration test")
        # response.content is the binary audio data
        assert len(response.content) > 1000

    def test_audio_transcription(self, client, tmp_path):
        # Generate audio first using TTS
        audio_data = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This is a test transcription."
        ).content

        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(audio_data)

        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(model="stt-model", file=f)
        assert "test" in transcription.text.lower()


@pytest.mark.integration
@pytest.mark.diffusers
class TestImage:
    """End-to-end image generation, editing and variations through the
    diffusers loader. One sdxl-turbo deployment backs all three endpoints
    (text2img + img2img + inpaint, weight-shared via from_pipe). The generated
    image is reused as the input for the edit and variation calls."""

    SIZE = "512x512"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("image-model")

    @staticmethod
    def _decode(b64: str) -> bytes:
        return base64.b64decode(b64)

    @staticmethod
    def _assert_png(data: bytes, *, expect_size: tuple[int, int] | None = None) -> None:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.verify()
        assert img.format == "PNG"
        if expect_size is not None:
            assert img.size == expect_size

    def test_image_generation(self, client):
        response = client.images.generate(
            model="image-model", prompt="a red apple on a table", size=self.SIZE, response_format="b64_json"
        )
        assert response.data[0].b64_json
        self._assert_png(self._decode(response.data[0].b64_json), expect_size=(512, 512))

    def test_image_edit(self, client, tmp_path):
        source = client.images.generate(
            model="image-model", prompt="a plain blue sky", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            edited = client.images.edit(
                model="image-model",
                image=f,
                prompt="a blue sky with a bright sun",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(512, 512))

    def test_image_edit_with_mask(self, client, tmp_path):
        from PIL import Image

        source = client.images.generate(
            model="image-model", prompt="a grassy field", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        # White rectangle marks the region to repaint; the rest is preserved.
        mask = Image.new("RGB", (512, 512), (0, 0, 0))
        for x in range(128, 384):
            for y in range(128, 384):
                mask.putpixel((x, y), (255, 255, 255))
        mask_path = tmp_path / "mask.png"
        mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            edited = client.images.edit(
                model="image-model",
                image=img_f,
                mask=mask_f,
                prompt="a grassy field with a small red flower",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(512, 512))

    def test_image_variation(self, client, tmp_path):
        source = client.images.generate(
            model="image-model", prompt="a yellow lemon", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            variation = client.images.create_variation(
                model="image-model", image=f, size=self.SIZE, response_format="b64_json"
            )
        assert variation.data[0].b64_json
        self._assert_png(self._decode(variation.data[0].b64_json), expect_size=(512, 512))


@pytest.mark.integration
@pytest.mark.stable_diffusion_cpp
class TestImageStableDiffusionCpp:
    """End-to-end CPU image generation, editing and variations through the
    stable_diffusion_cpp loader. One single-file SD2.1 GGUF deployment backs all
    three endpoints (txt2img + img2img + mask). The generated image is reused as
    the input for the edit and variation calls. Small size + few steps keep the
    CPU run tractable; the assertions check a valid PNG of the requested size,
    not image quality."""

    SIZE = "256x256"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("image-cpu-model")

    @staticmethod
    def _decode(b64: str) -> bytes:
        return base64.b64decode(b64)

    @staticmethod
    def _assert_png(data: bytes, *, expect_size: tuple[int, int] | None = None) -> None:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.verify()
        assert img.format == "PNG"
        if expect_size is not None:
            assert img.size == expect_size

    def test_image_generation(self, client):
        response = client.images.generate(
            model="image-cpu-model", prompt="a red apple on a table", size=self.SIZE, response_format="b64_json"
        )
        assert response.data[0].b64_json
        self._assert_png(self._decode(response.data[0].b64_json), expect_size=(256, 256))

    def test_image_edit(self, client, tmp_path):
        source = client.images.generate(
            model="image-cpu-model", prompt="a plain blue sky", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            edited = client.images.edit(
                model="image-cpu-model",
                image=f,
                prompt="a blue sky with a bright sun",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(256, 256))

    def test_image_edit_with_mask(self, client, tmp_path):
        from PIL import Image

        source = client.images.generate(
            model="image-cpu-model", prompt="a grassy field", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        # White rectangle marks the region to repaint; the rest is preserved.
        mask = Image.new("RGB", (256, 256), (0, 0, 0))
        for x in range(64, 192):
            for y in range(64, 192):
                mask.putpixel((x, y), (255, 255, 255))
        mask_path = tmp_path / "mask.png"
        mask.save(mask_path, format="PNG")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            edited = client.images.edit(
                model="image-cpu-model",
                image=img_f,
                mask=mask_f,
                prompt="a grassy field with a small red flower",
                size=self.SIZE,
                response_format="b64_json",
            )
        assert edited.data[0].b64_json
        self._assert_png(self._decode(edited.data[0].b64_json), expect_size=(256, 256))

    def test_image_variation(self, client, tmp_path):
        source = client.images.generate(
            model="image-cpu-model", prompt="a yellow lemon", size=self.SIZE, response_format="b64_json"
        )
        image_path = tmp_path / "source.png"
        image_path.write_bytes(self._decode(source.data[0].b64_json))

        with open(image_path, "rb") as f:
            variation = client.images.create_variation(
                model="image-cpu-model", image=f, size=self.SIZE, response_format="b64_json"
            )
        assert variation.data[0].b64_json
        self._assert_png(self._decode(variation.data[0].b64_json), expect_size=(256, 256))


# Responses tools use the *flattened* function shape (name/parameters at the
# top level), unlike chat completions which nests them under "function".
_WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesEndpoint:
    """End-to-end /v1/responses over the vLLM chat pipeline. Verifies the official
    OpenAI SDK's ``responses.create`` parses our payload and that unsupported
    features are rejected, not silently dropped."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_basic_response_parses_with_openai_sdk(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="Say hello in one word.",
            max_output_tokens=20,
        )
        assert resp.object == "response"
        assert resp.status in {"completed", "incomplete"}
        assert resp.output_text.strip()
        assert resp.usage.input_tokens > 0
        # Stored by default (OpenAI parity), so an unset `store` echoes True.
        assert resp.store is True

    def test_instructions_and_message_list_input(self, client):
        resp = client.responses.create(
            model="chat-capable",
            instructions="Answer in exactly one word.",
            input=[{"role": "user", "content": "What color is a clear daytime sky?"}],
            max_output_tokens=20,
        )
        assert resp.output_text.strip()

    def test_tool_call_emits_function_call_item(self, client):
        resp = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
        )
        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert function_calls, f"expected a function_call output item, got {[i.type for i in resp.output]}"
        assert function_calls[0].name == "get_weather"
        assert "Paris" in function_calls[0].arguments

    def test_streaming_emits_event_protocol(self, client):
        # stream=True drives the chat pipeline in streaming mode and translates
        # its chunks into the Responses event protocol. The official SDK parses
        # the named events and reconstructs the final response.
        stream = client.responses.create(
            model="chat-capable",
            input="Say hello in one word.",
            max_output_tokens=20,
            stream=True,
        )
        types: list[str] = []
        text_deltas: list[str] = []
        completed = None
        for event in stream:
            types.append(event.type)
            if event.type == "response.output_text.delta":
                text_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response

        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        assert "".join(text_deltas).strip(), "expected streamed output text"
        assert completed is not None
        assert completed.status in {"completed", "incomplete"}
        # The streamed deltas must reconstruct the final message text.
        assert "".join(text_deltas).strip() == completed.output_text.strip()
        assert completed.usage.input_tokens > 0

    def test_streaming_tool_call_emits_argument_deltas(self, client):
        stream = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
            stream=True,
        )
        arg_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                arg_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert completed is not None
        function_calls = [item for item in completed.output if item.type == "function_call"]
        assert function_calls, f"expected a function_call item, got {[i.type for i in completed.output]}"
        assert function_calls[0].name == "get_weather"
        # streamed argument fragments must reconstruct the final arguments
        assert "".join(arg_deltas) == function_calls[0].arguments

    def test_truncation_reports_incomplete_details(self, client):
        # A generation cut short by max_output_tokens is `incomplete` with a reason,
        # not `completed` — the only signal a client has that output was truncated.
        resp = client.responses.create(
            model="chat-capable",
            input="Write a long essay about the sea.",
            max_output_tokens=16,
        )
        assert resp.status == "incomplete"
        assert resp.incomplete_details is not None
        assert resp.incomplete_details.reason == "max_output_tokens"
        assert resp.output_text.strip()

    def test_unknown_previous_response_id_404(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": "resp_does_not_exist"},
            timeout=60,
        )
        assert response.status_code == 404, response.text

    def test_background_rejected_400(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "background": True},
            timeout=60,
        )
        assert response.status_code == 400, response.text
        assert "background" in response.json()["error"]["message"]

    def test_hosted_tool_rejected_400(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "search the web", "tools": [{"type": "web_search"}]},
            timeout=60,
        )
        assert response.status_code == 400, response.text
        assert "hosted tool" in response.json()["error"]["message"]


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesState:
    """Server-side conversation state on /v1/responses, end-to-end against the real
    store (``memory://`` — a detached Ray actor on the live cluster).

    The payoff test is `test_continuation_recalls_earlier_turn`: the model answers from
    history the client never resent, which is the whole point of the endpoint. The rest
    pin the lifecycle (store/retrieve/delete) and the failure modes that must not
    silently degrade.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-capable")

    def test_continuation_recalls_earlier_turn(self, client):
        first = client.responses.create(
            model="chat-capable",
            input="My name is Alex. Remember it.",
            max_output_tokens=30,
        )
        assert first.store is True

        # Turn 2 sends only the new question — the name is recalled from stored state.
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        assert second.previous_response_id == first.id
        assert "alex" in second.output_text.lower()

    def test_continuation_chains_across_three_turns(self, client):
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="I live in Berlin.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        third = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=second.id,
            max_output_tokens=20,
        )
        # Turn 1's fact survives two hops — each snapshot embeds the whole conversation.
        assert "alex" in third.output_text.lower()

    def test_streaming_response_can_be_continued(self, client):
        stream = client.responses.create(
            model="chat-capable",
            input="My name is Alex. Remember it.",
            max_output_tokens=30,
            stream=True,
        )
        completed = None
        for event in stream:
            if event.type == "response.completed":
                completed = event.response
        assert completed is not None
        assert completed.store is True

        # A streamed response is persisted by re-reading its terminal event, so this
        # proves that path stores the same shape the non-streaming one does.
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=completed.id,
            max_output_tokens=20,
        )
        assert "alex" in second.output_text.lower()

    def test_get_returns_stored_response(self, client):
        created = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20)
        fetched = client.responses.retrieve(created.id)
        assert fetched.id == created.id
        assert fetched.output_text == created.output_text

    def test_input_items_lists_what_went_in(self):
        created = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hi.", "max_output_tokens": 20},
            timeout=120,
        ).json()
        listed = httpx.get(f"{OPENAI_API_BASE}/responses/{created['id']}/input_items", timeout=60)
        assert listed.status_code == 200, listed.text
        body = listed.json()
        assert body["object"] == "list"
        assert any("Say hi." in str(item.get("content", "")) for item in body["data"])

    def test_input_items_reflects_the_continued_chain(self, client):
        # After a continuation the snapshot's input is the resolved history, not just
        # the turn the client sent: user -> assistant -> user, in order.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="What is my name?",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        body = httpx.get(f"{OPENAI_API_BASE}/responses/{second.id}/input_items", timeout=60).json()
        roles = [item.get("role") for item in body["data"]]
        assert roles == ["user", "assistant", "user"], body["data"]
        assert "My name is Alex." in str(body["data"][0]["content"])

    def test_store_false_is_not_retrievable(self):
        created = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "Say hi.", "max_output_tokens": 20, "store": False},
            timeout=120,
        ).json()
        assert created["store"] is False
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{created['id']}", timeout=60).status_code == 404

    def test_streamed_store_false_is_not_retrievable(self, client):
        # The streaming path persists by re-reading its own terminal event, so
        # `store: false` has to suppress a different branch than the non-streaming one.
        stream = client.responses.create(
            model="chat-capable",
            input="Say hi.",
            max_output_tokens=20,
            stream=True,
            store=False,
        )
        completed = None
        for event in stream:
            if event.type == "response.completed":
                completed = event.response
        assert completed is not None
        assert completed.store is False
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{completed.id}", timeout=60).status_code == 404

    def test_store_false_still_reads_history(self, client):
        # store=false governs writing this turn, not reading the chain: a caller can
        # continue a conversation without adding to it.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=first.id,
            max_output_tokens=20,
            store=False,
        )
        assert "alex" in second.output_text.lower()
        assert httpx.get(f"{OPENAI_API_BASE}/responses/{second.id}", timeout=60).status_code == 404

    def test_previous_response_id_of_unstored_response_404s(self, client):
        # An id that was never persisted is indistinguishable from an unknown one.
        unstored = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20, store=False)
        continued = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": unstored.id},
            timeout=60,
        )
        assert continued.status_code == 404, continued.text

    def test_continuation_survives_parent_deletion(self, client):
        # Each snapshot embeds the whole conversation, so a chain is not a linked list:
        # deleting a parent must not strand its children.
        first = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)
        second = client.responses.create(
            model="chat-capable",
            input="I live in Berlin.",
            previous_response_id=first.id,
            max_output_tokens=20,
        )
        assert httpx.delete(f"{OPENAI_API_BASE}/responses/{first.id}", timeout=60).status_code == 200

        third = client.responses.create(
            model="chat-capable",
            input="What is my name? Reply with just the name.",
            previous_response_id=second.id,
            max_output_tokens=20,
        )
        assert "alex" in third.output_text.lower()

    def test_concurrent_branches_from_one_parent_are_independent(self, client):
        # Several turns may fan out from the same previous_response_id; each must get
        # its own id and its own snapshot rather than racing over shared state.
        parent = client.responses.create(model="chat-capable", input="My name is Alex.", max_output_tokens=20)

        def _branch(_: int):
            return client.responses.create(
                model="chat-capable",
                input="What is my name? Reply with just the name.",
                previous_response_id=parent.id,
                max_output_tokens=20,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            branches = list(pool.map(_branch, range(5)))

        assert len({b.id for b in branches}) == 5, "branch ids collided"
        for branch in branches:
            assert branch.previous_response_id == parent.id
            assert "alex" in branch.output_text.lower()

    def test_delete_then_get_and_continue_both_404(self, client):
        created = client.responses.create(model="chat-capable", input="Say hi.", max_output_tokens=20)

        deleted = httpx.delete(f"{OPENAI_API_BASE}/responses/{created.id}", timeout=60)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"id": created.id, "object": "response", "deleted": True}

        assert httpx.get(f"{OPENAI_API_BASE}/responses/{created.id}", timeout=60).status_code == 404
        # A deleted conversation is gone for continuation too, not just retrieval.
        continued = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": created.id},
            timeout=60,
        )
        assert continued.status_code == 404, continued.text

    def test_function_call_output_round_trip(self, client):
        # The stateful tool loop: the client returns only the tool result, and the
        # pending call it answers is recovered from stored history.
        first = client.responses.create(
            model="chat-capable",
            input="What is the weather in Paris?",
            tools=[_WEATHER_TOOL_RESPONSES],
            tool_choice="required",
            max_output_tokens=128,
        )
        calls = [item for item in first.output if item.type == "function_call"]
        assert calls, f"expected a function_call item, got {[i.type for i in first.output]}"

        second = client.responses.create(
            model="chat-capable",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": json.dumps({"temp_c": 18, "sky": "rain"}),
                }
            ],
            tools=[_WEATHER_TOOL_RESPONSES],
            previous_response_id=first.id,
            max_output_tokens=128,
        )
        assert "18" in second.output_text, second.output_text

    def test_get_unknown_id_404(self):
        assert httpx.get(f"{OPENAI_API_BASE}/responses/resp_does_not_exist", timeout=60).status_code == 404

    def test_delete_unknown_id_404(self):
        assert httpx.delete(f"{OPENAI_API_BASE}/responses/resp_does_not_exist", timeout=60).status_code == 404

    def test_malformed_id_404_and_leaves_state_intact(self):
        # response_id is a state-key segment; a traversal-shaped id must not resolve to
        # (or delete) anything else the store holds.
        resp = httpx.request("DELETE", f"{OPENAI_API_BASE}/responses/..%2F..%2Feffective%2Fmodelship%20api", timeout=60)
        assert resp.status_code == 404, resp.text
        # The gateway's own effective config is untouched — /v1/models still answers.
        assert httpx.get(f"{OPENAI_API_BASE}/models", timeout=60).status_code == 200


@pytest.mark.integration
@pytest.mark.vllm
class TestResponsesReasoning:
    """Reasoning surfaces as a first-class ``reasoning`` output item on
    /v1/responses (its spec-correct home), distinct from the off-spec
    ``message.reasoning`` field on chat completions."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-reasoning")

    def test_reasoning_output_item_present(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-reasoning", "input": "Briefly: what is 7 times 8?", "max_output_tokens": 512},
            timeout=120,
        )
        assert response.status_code == 200, response.text
        output = response.json()["output"]

        reasoning_items = [item for item in output if item["type"] == "reasoning"]
        assert reasoning_items, f"expected a reasoning output item, got {[i['type'] for i in output]}"
        summary_text = "".join(s["text"] for item in reasoning_items for s in item.get("summary", []))
        assert summary_text.strip(), "expected non-empty reasoning summary text"
        # `<think>` markers must be stripped before reshaping into the item.
        assert "<think>" not in summary_text

        message_items = [item for item in output if item["type"] == "message"]
        assert message_items, "expected an assistant message output item alongside reasoning"

    def test_streaming_emits_reasoning_summary_deltas(self, client):
        stream = client.responses.create(
            model="chat-reasoning",
            input="Briefly: what is 7 times 8?",
            max_output_tokens=512,
            stream=True,
        )
        reasoning_deltas: list[str] = []
        completed = None
        for event in stream:
            if event.type == "response.reasoning_summary_text.delta":
                reasoning_deltas.append(event.delta)
            elif event.type == "response.completed":
                completed = event.response
        assert reasoning_deltas, "expected streamed reasoning summary deltas"
        assert "<think>" not in "".join(reasoning_deltas)
        assert completed is not None
        reasoning_items = [item for item in completed.output if item.type == "reasoning"]
        assert reasoning_items, "expected a reasoning output item in the completed response"


def _running_replicas(model_name: str) -> int:
    """Count RUNNING replicas of the deployment serving `model_name`, read from
    the Serve REST status API. The app name is `<model_name>-<fingerprint>`, so
    match by prefix; the single inner deployment carries the replica list."""
    resp = httpx.get(SERVE_STATUS_URL, timeout=10)
    resp.raise_for_status()
    apps = resp.json().get("applications", {})
    for app_name, app in apps.items():
        if app_name == model_name or app_name.startswith(f"{model_name}-"):
            for dep in app.get("deployments", {}).values():
                return sum(1 for r in dep.get("replicas", []) if r.get("state") == "RUNNING")
    return 0


def _wait_for_replicas(model_name: str, predicate, deadline_s: float) -> int:
    """Poll replica count until `predicate(count)` holds or the deadline passes.
    Returns the last observed count either way (caller asserts)."""
    end = time.time() + deadline_s
    count = _running_replicas(model_name)
    while time.time() < end:
        count = _running_replicas(model_name)
        if predicate(count):
            return count
        time.sleep(2)
    return count


def _hammer(client: OpenAI, model: str, stop: threading.Event, errors: list) -> None:
    """Keep one request in flight at a time until `stop` is set. Several of these
    running concurrently sustain enough load to push past the autoscaler's
    per-replica setpoint."""
    while not stop.is_set():
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Write a long, detailed story about a curious robot."}],
                max_tokens=256,
            )
        except Exception as exc:
            # Surfaced via the shared list, not raised in the worker thread.
            errors.append(exc)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.autoscaling
class TestAutoscaling:
    """End-to-end check that a model's autoscaling_config actually drives Ray
    Serve: replicas scale out under sustained concurrent load (bounded by
    max_replicas) and scale back to min_replicas once the load stops."""

    MODEL = "autoscale-llama"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy(self.MODEL)

    def test_scales_out_under_load_then_back_to_min(self, client):
        # Idle baseline: the deployment sits at min_replicas (1).
        baseline = _wait_for_replicas(self.MODEL, lambda n: n == 1, deadline_s=60)
        assert baseline == 1, f"expected to start at min_replicas=1, saw {baseline}"

        stop = threading.Event()
        errors: list[Exception] = []
        # 8 concurrent in-flight requests vs target_ongoing_requests=1 asks the
        # autoscaler for ~8 replicas, capped at max_replicas=3.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for _ in range(8):
                pool.submit(_hammer, client, self.MODEL, stop, errors)
            try:
                # Autoscaler needs a look-back window of load metrics; allow generous time.
                peak = _wait_for_replicas(self.MODEL, lambda n: n > 1, deadline_s=120)
            finally:
                stop.set()

        assert peak > 1, f"expected scale-out under load, replicas stayed at {peak}"
        assert peak <= 3, f"replicas {peak} exceeded max_replicas=3"
        assert not errors, f"load requests errored during scale-out: {errors[:3]}"

        # Load stopped: scale back in to min_replicas within the downscale window + slack.
        settled = _wait_for_replicas(self.MODEL, lambda n: n == 1, deadline_s=180)
        assert settled == 1, f"expected scale-in to min_replicas=1 after load, saw {settled}"


def _model_in_all_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    """True iff `model` appears in /v1/models on EVERY sampled request. With the
    gateway load-balanced across replicas, a stale replica would omit it on some
    requests — so 'present in all samples' means every replica has converged."""
    return all(model in {m.id for m in client.models.list().data} for _ in range(samples))


def _model_in_no_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    return all(model not in {m.id for m in client.models.list().data} for _ in range(samples))


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.gateway_ha
class TestGatewayReplicaConsistency:
    """With 2 gateway replicas (the session starts --gateway-replicas 2), a deployed
    model must become routable on BOTH replicas and a removed one must stop routing
    on both — i.e. the coordinator watch loop reconciles every replica, not just the
    one a direct push would have hit."""

    def test_add_and_remove_propagate_to_all_replicas(self, client, model_deployer):
        # Warm both replicas (spread requests so each starts its watch loop).
        for _ in range(10):
            client.models.list()

        # Deploy: the model becomes routable on every replica.
        model_deployer.deploy("chat-llama-server-plain")
        assert _poll(lambda: _model_in_all_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "deployed model did not become routable on all gateway replicas"
        )
        completion = client.chat.completions.create(
            model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
        assert completion.choices[0].message.content is not None

        # Reconcile to a different model — chat-llama-server-plain is removed everywhere.
        model_deployer.deploy("chat-llama-server")
        assert _poll(lambda: _model_in_no_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "removed model still routable on some gateway replica"
        )

        # Requests to the removed model now 404 on every replica — none route into
        # the torn-down deployment (which would surface as a 5xx, not a 404).
        import openai

        for _ in range(20):
            with pytest.raises(openai.NotFoundError):
                client.chat.completions.create(
                    model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )
