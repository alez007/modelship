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
    "chat-limited": {
        "name": "chat-limited",
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_cpp",
        "num_cpus": 1,
        "llama_cpp_config": {
            "tool_calls_enabled": False,
        },
    },
    "chat-llama-mship": {
        "name": "chat-llama-mship",
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_cpp",
        "num_cpus": 1,
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
        "loader": "llama_cpp",
        "num_cpus": 1,
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 1,
            "upscale_delay_s": 2,
            "downscale_delay_s": 10,
        },
    },
    "chat-llama-reasoning": {
        "name": "chat-llama-reasoning",
        # Qwen3-0.6B in GGUF form: same family as the vLLM `chat-reasoning`
        # deployment (which uses the safetensors checkpoint), so the
        # model emits `<think>...</think>` and supports Hermes-style tool
        # calls in the same chat template. Lets us exercise reasoning,
        # tools, and reasoning+tools through the llama_cpp loader in one
        # deployment. Reasoning chains need headroom, so n_ctx is bumped.
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_cpp",
        "num_cpus": 1,
        "llama_cpp_config": {
            "n_ctx": 4096,
        },
    },
    "chat-transformers": {
        "name": "chat-transformers",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "usecase": "generate",
        "loader": "transformers",
        "num_cpus": 2,
        "transformers_config": {
            "device": "cpu",
            "torch_dtype": "float32",
        },
    },
    "chat-transformers-llama3-json": {
        "name": "chat-transformers-llama3-json",
        # Llama-3.1-8B-Instruct emits the JSON-format tool call defined by
        # Meta's spec — bare ``{"name": "...", "parameters": {...}}``. The
        # chat template references ``<|python_tag|>``, so auto-detection
        # resolves to ``llama3_json``. 8B is the smallest Llama-3.x
        # variant Meta certifies for tool / function calling — Llama-3.2-1B
        # has the same chat template but does not reliably emit the
        # JSON-call shape on ``tool_choice="auto"``. Like the Mistral-7B
        # transformers deployment, 8B in float32 is too memory-heavy for
        # CPU+float32, so this is the second exception pinned to a full
        # GPU (``device=cuda``, native bfloat16 via ``torch_dtype="auto"``).
        # Requires ``HF_TOKEN`` for the gated repo.
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "usecase": "generate",
        "loader": "transformers",
        "num_cpus": 5,
        "num_gpus": 1,
        "transformers_config": {
            "device": "cuda",
        },
    },
    "chat-transformers-mistral": {
        "name": "chat-transformers-mistral",
        # Mistral-7B-Instruct-v0.3 emits ``[TOOL_CALLS]`` followed by a JSON
        # array of tool calls. The marker is registered as a *special added
        # token* on the tokenizer, so by default ``skip_special_tokens=True``
        # in ``TextIteratorStreamer`` would strip it before the parser sees
        # it. The fix in this PR detects this case via
        # ``markers_are_specials`` on the parser, pins
        # ``_resolved_skip_special_tokens=False`` on the model config, and
        # the transformers loader flips both ``TextIteratorStreamer`` and
        # the non-streaming ``pipeline()`` call accordingly.
        #
        # 7B is the smallest official Mistral that ships the v3+ tool
        # protocol; the rest of the transformers suite runs on CPU+float32
        # but a 7B model in float32 (~28GB) creates too much memory
        # pressure on the integration host, so this deployment is the
        # exception — pinned to a full GPU. ``torch_dtype`` is left at the
        # ``"auto"`` default which picks the model's native bfloat16.
        # Requires ``HF_TOKEN`` (gated repo).
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "usecase": "generate",
        "loader": "transformers",
        "num_cpus": 5,
        "num_gpus": 1,
        "transformers_config": {
            "device": "cuda",
        },
    },
    "chat-transformers-reasoning": {
        "name": "chat-transformers-reasoning",
        # Qwen3-0.6B safetensors — same family as the vLLM `chat-reasoning`
        # and llama_cpp `chat-llama-reasoning` deployments. Lets us
        # exercise reasoning, tools, and reasoning+tools through the
        # transformers loader's `ChatOutputStreamer` wiring on real model
        # output. CPU-only and float32 to match the rest of the
        # transformers integration suite.
        "model": "Qwen/Qwen3-0.6B",
        "usecase": "generate",
        "loader": "transformers",
        "num_cpus": 2,
        "transformers_config": {
            "device": "cpu",
            "torch_dtype": "float32",
        },
    },
    "chat-transformers-function-gemma": {
        "name": "chat-transformers-function-gemma",
        # Google's FunctionGemma (Gemma 2 family) uses the `<start_function_call>`
        # and `<escape>` syntax. It's a 270M parameter model, small enough
        # to run on CPU+float32.
        "model": "google/functiongemma-270m-it",
        "usecase": "generate",
        "loader": "transformers",
        "num_cpus": 2,
        "transformers_config": {
            "device": "cpu",
            "torch_dtype": "float32",
        },
    },
    "embed-model": {
        "name": "embed-model",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "usecase": "embed",
        "loader": "transformers",
        "num_cpus": 1,
    },
    "stt-model": {
        "name": "stt-model",
        "model": "openai/whisper-tiny",
        "usecase": "transcription",
        "loader": "transformers",
        "num_cpus": 1,
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
        # A per-session file:// state store, shared with the operator (see
        # mship_cluster). The default is now memory://, which is per-process — but
        # each reconcile runs in its OWN subprocess and must read the effective set
        # the prior deploy wrote to tear down models that dropped out, so the
        # integration tests opt into a shared cross-process file store under tmp.
        self._state_store = f"file://{tmp_dir / 'state'}"
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
                    "--state-store",
                    self._state_store,
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
    # Per-session file:// state store, shared with every _Deployer reconcile via
    # --state-store. memory:// (the default) is per-process, so reconcile in a
    # separate subprocess couldn't read this operator's effective set; a shared
    # file store under tmp keeps that cross-process sharing without touching the
    # /.cache/state default.
    state_store = f"file://{tmp_dir / 'state'}"

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
            "--state-store",
            state_store,
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
class TestChatTransformers:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-transformers")

    def test_tool_calling_transformers_loader(self, client):
        """Round-trip a Hermes-style tool call through the transformers loader.

        Uses the same Qwen2.5-0.5B-Instruct weights as the vLLM `chat-capable`
        deployment but goes through the modelship-side tool-calling toolkit
        (apply_chat_template(tools=...) on input, hermes parser on output).
        """
        completion = client.chat.completions.create(
            model="chat-transformers",
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

    def test_tool_calling_streaming_transformers_loader(self, client):
        """Stream a tool call through the transformers loader and verify the
        delta sequence matches the OpenAI streaming contract.

        Asserts:
        - the function name arrives in exactly one delta;
        - arguments arrive across multiple deltas (incremental, not buffered);
        - concatenated arguments form valid JSON containing the expected key;
        - the final delta carries ``finish_reason="tool_calls"``.
        """
        stream = client.chat.completions.create(
            model="chat-transformers",
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
        assert call_0["id"], "expected an id on the first tool-call delta"
        assert call_0["name"] == "get_weather"
        # Name must be sent exactly once (not on every delta).
        assert collected["name_deltas"] == 1, f"expected one name delta, got {collected['name_deltas']}"
        # Arguments must arrive incrementally across multiple deltas — that's the
        # whole point of switching from block-level buffering to vLLM-style
        # diff streaming. Exact count depends on the model, but it must be > 1.
        assert collected["args_deltas"] >= 2, (
            f"expected arguments to stream incrementally, got {collected['args_deltas']} args delta(s)"
        )
        # Concatenated args must form valid JSON containing the city.
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"


@pytest.mark.integration
@pytest.mark.llama_cpp
class TestChatLlamaCpp:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-mship")

    def test_tool_calling_llama_cpp_loader(self, client):
        """Round-trip a Hermes-style tool call through the llama_cpp loader.

        Same Qwen2.5-0.5B-Instruct weights as `chat-capable` (vLLM) and
        `chat-transformers`, but in GGUF form via llama-cpp-python. Auto-detected
        hermes parser renders the prompt with `tools=...` and parses the
        `<tool_call>{...}</tool_call>` markers out of raw completion output.
        """
        completion = client.chat.completions.create(
            model="chat-llama-mship",
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

    def test_tool_calling_streaming_llama_cpp_loader(self, client):
        """Stream a tool call through the llama_cpp loader and verify the
        delta sequence matches the OpenAI streaming contract.

        Same shape as `test_tool_calling_streaming_transformers_loader` —
        asserts a single name delta, multiple incremental argument deltas,
        valid JSON on concatenation, and final ``finish_reason="tool_calls"``.
        """
        stream = client.chat.completions.create(
            model="chat-llama-mship",
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
        assert call_0["id"], "expected an id on the first tool-call delta"
        assert call_0["name"] == "get_weather"
        assert collected["name_deltas"] == 1, f"expected one name delta, got {collected['name_deltas']}"
        assert collected["args_deltas"] >= 2, (
            f"expected arguments to stream incrementally, got {collected['args_deltas']} args delta(s)"
        )
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"

    def test_response_format_json_object_constrains_unprompted_output(self, client):
        """Prompt is natural-language; the grammar constraint produces JSON."""
        completion = client.chat.completions.create(
            model="chat-llama-mship",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        content = completion.choices[0].message.content
        assert content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

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
            model="chat-llama-mship",
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
            model="chat-llama-mship",
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
            model="chat-llama-mship",
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

    def test_response_format_with_active_tools_rejected_by_gateway(self):
        """Protocol-layer validator rejects tools + response_format when
        tool_choice is anything other than 'none'. The schema grammar would
        block tool-call markers from being emitted, so we surface the conflict
        upfront rather than silently breaking tool calling.
        """
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-mship",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": "auto",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "unused",
                        "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                        "strict": True,
                    },
                },
            },
            timeout=30,
        )
        assert response.status_code in (400, 422), (
            f"expected client-error status, got {response.status_code}: {response.text}"
        )
        assert "tool_choice='none'" in response.text


@pytest.mark.integration
@pytest.mark.llama_cpp
class TestChatLlamaCppReasoning:
    """End-to-end reasoning + tool calling through the llama_cpp loader.

    Same Qwen3-0.6B family as the vLLM `chat-reasoning` deployment but
    in GGUF form, so the modelship-side ``ChatOutputStreamer`` is
    actually exercised on real model output (vLLM has its own native
    reasoning parser, llama_cpp does not). One deployment covers
    three scenarios because Qwen3 emits ``<think>...</think>`` AND
    supports Hermes-style tool calls in the same chat template.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-reasoning")

    def test_reasoning_completion_llama_cpp(self):
        """Non-streaming: ``<think>...`` block routes to ``message.reasoning``,
        the final answer lands in ``message.content``, no marker leakage."""
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-reasoning",
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

    def test_reasoning_streaming_llama_cpp(self):
        """Streaming: at least one delta carries ``reasoning``; concatenated
        reasoning is non-empty; markers never leak into either field."""
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-reasoning",
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

    def test_reasoning_with_tools_llama_cpp(self, client):
        """Reasoning + tool calling in one round-trip.

        Asserts that when a reasoning model also emits a tool call, the
        single-pass ``ChatOutputStreamer`` populates BOTH
        ``message.reasoning`` and ``message.tool_calls``, and that
        ``finish_reason="tool_calls"``.
        """
        completion = client.chat.completions.create(
            model="chat-llama-reasoning",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        # The OpenAI Python SDK exposes unknown fields via ``model_extra``.
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert "<think>" not in reasoning
        tool_calls = message.tool_calls
        assert tool_calls, f"expected a tool call, got content={message.content!r}, reasoning={reasoning!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_tool_markers_inside_reasoning_not_double_counted(self, client):
        """Tool-call markers emitted *inside* ``<think>`` must route to
        reasoning, never become real tool calls.

        Coaxes the model into illustrating tool-call syntax inside its
        reasoning and then making one actual call. The single-pass
        ``ChatOutputStreamer`` must:

        - Surface the illustrative ``<tool_call>...</tool_call>`` text
          inside ``message.reasoning`` (proving it was treated as
          reasoning bytes, not a real call).
        - Emit exactly ONE ``tool_calls`` entry (the actual post-
          reasoning call), not multiples.

        Real models are non-deterministic; if the prompt fails to
        produce literal markers in reasoning, we skip the
        marker-routing assertion rather than flake — the
        single-tool-call assertion still has value either way. The
        deterministic equivalent is exercised in unit tests
        (``tests/test_reasoning.py::TestComposition``).
        """
        completion = client.chat.completions.create(
            model="chat-llama-reasoning",
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

        # Hard assertions: regardless of prompt compliance, the streamer
        # must produce exactly one real tool call for the weather query.
        assert tool_calls, (
            f"expected exactly one real tool call, got content={message.content!r}, reasoning={reasoning!r}"
        )
        assert len(tool_calls) == 1, (
            f"expected exactly one tool call (markers inside <think> must not be double-counted); "
            f"got {len(tool_calls)} calls={[tc.function.name for tc in tool_calls]}"
        )
        assert tool_calls[0].function.name == "get_weather"
        assert completion.choices[0].finish_reason == "tool_calls"

        # Soft assertion: only meaningful if the model actually quoted
        # the marker syntax inside its reasoning.
        if "<tool_call>" in reasoning:
            # Reasoning carries the literal marker text — confirms the
            # streamer routed it to the reasoning view rather than
            # parsing it as a real call (which would have shown up as a
            # second tool_calls entry).
            assert "</tool_call>" in reasoning, (
                f"reasoning has an unmatched <tool_call> marker (open without close): {reasoning!r}"
            )

    def test_response_format_with_reasoning_deployment_rejected(self):
        """A JSON grammar would exclude the `<` token, breaking the reasoning
        parser's `<think>...` emission. The loader rejects the combination at
        request time rather than producing malformed output.
        """
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-llama-reasoning",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "response_format": {
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
                "max_tokens": 256,
            },
            timeout=60,
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"
        assert "reasoning" in response.text.lower()


@pytest.mark.integration
class TestChatTransformersLlama3Json:
    """End-to-end llama3_json tool calling through the transformers loader.

    Llama-3.1-8B-Instruct on GPU emits the JSON-format tool call defined
    by Meta's spec — bare ``{"name": "...", "parameters": {...}}``. The
    auto-detector picks ``llama3_json`` from the chat template's
    ``<|python_tag|>`` reference. This class is the first end-to-end
    exercise of the ``llama3_json`` parser on real model output (vLLM
    has its own native parser; transformers/llama_cpp run through the
    cross-loader registry).
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-transformers-llama3-json")

    def test_tool_calling_transformers_llama3_json_loader(self, client):
        """Round-trip a bare-JSON tool call through the transformers loader.

        Verifies the parser surfaces ``message.tool_calls`` from a
        ``{"name": "...", "parameters": {...}}`` response and canonicalizes
        the ``parameters`` field bytes into ``arguments`` for the OpenAI
        contract.
        """
        completion = client.chat.completions.create(
            model="chat-transformers-llama3-json",
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

    def test_tool_calling_streaming_transformers_llama3_json_loader(self, client):
        """Stream a bare-JSON tool call and verify the OpenAI delta contract.

        Same shape as the Hermes/Mistral streaming tests: exactly one name
        delta, multiple incremental argument deltas, valid JSON on
        concatenation, final ``finish_reason="tool_calls"``.
        """
        stream = client.chat.completions.create(
            model="chat-transformers-llama3-json",
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
        assert call_0["id"], "expected an id on the first tool-call delta"
        assert call_0["name"] == "get_weather"
        assert collected["name_deltas"] == 1, f"expected one name delta, got {collected['name_deltas']}"
        assert collected["args_deltas"] >= 2, (
            f"expected arguments to stream incrementally, got {collected['args_deltas']} args delta(s)"
        )
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"


@pytest.mark.integration
class TestChatTransformersMistral:
    """End-to-end Mistral tool calling through the transformers loader.

    Regression coverage for the ``markers_are_specials`` fix. Mistral
    tokenizers register ``[TOOL_CALLS]`` as a special added token, so
    the default ``TextIteratorStreamer(skip_special_tokens=True)`` would
    strip the marker before the parser sees it — the parser would
    silently miss every tool call. The fix pins
    ``_resolved_skip_special_tokens=False`` from the parser flag and
    the transformers loader honors it (plus a streamer-side noise
    stripper for the other specials that now leak through).

    If this class fails with empty ``tool_calls`` and the marker text
    visible in ``message.content``, the loader plumbing has regressed.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-transformers-mistral")

    def test_tool_calling_transformers_mistral_loader(self, client):
        """Round-trip a ``[TOOL_CALLS]``-prefixed tool call through the
        transformers loader.

        Verifies the parser surfaces ``message.tool_calls`` from a
        ``[TOOL_CALLS][{"name": "...", "arguments": {...}}]`` response —
        which only works if the marker survived the tokenizer's
        special-token stripping.
        """
        completion = client.chat.completions.create(
            model="chat-transformers-mistral",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=128,
        )
        tool_calls = completion.choices[0].message.tool_calls
        assert tool_calls, (
            f"expected a tool call (Mistral [TOOL_CALLS] marker likely stripped before parser saw it); "
            f"got content={completion.choices[0].message.content!r}"
        )
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"
        # The marker itself must never leak into content.
        assert "[TOOL_CALLS]" not in (completion.choices[0].message.content or "")

    def test_tool_calling_streaming_transformers_mistral_loader(self, client):
        """Stream a Mistral tool call and verify the OpenAI delta contract.

        Same shape as the Hermes/llama3_json streaming tests: exactly one
        name delta, multiple incremental argument deltas, valid JSON on
        concatenation, final ``finish_reason="tool_calls"``. The marker
        must not leak into any content delta.
        """
        stream = client.chat.completions.create(
            model="chat-transformers-mistral",
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
        assert call_0["id"], "expected an id on the first tool-call delta"
        assert call_0["name"] == "get_weather"
        assert collected["name_deltas"] == 1, f"expected one name delta, got {collected['name_deltas']}"
        assert collected["args_deltas"] >= 2, (
            f"expected arguments to stream incrementally, got {collected['args_deltas']} args delta(s)"
        )
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"
        assert "[TOOL_CALLS]" not in collected["content"]


@pytest.mark.integration
class TestChatTransformersReasoning:
    """End-to-end reasoning + tool calling through the transformers loader.

    Qwen3-0.6B safetensors driven through the HF text-generation pipeline.
    Verifies that ``transformers/openai/serving_chat.py`` plumbs
    ``_resolved_reasoning_parser`` into the unified ``ChatOutputStreamer``
    and that reasoning, tools, and reasoning+tools all surface correctly
    on real model output.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-transformers-reasoning")

    def test_reasoning_completion_transformers(self):
        """Non-streaming: ``<think>...`` block routes to ``message.reasoning``,
        the final answer lands in ``message.content``, no marker leakage."""
        response = httpx.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-transformers-reasoning",
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

    def test_reasoning_streaming_transformers(self):
        """Streaming: at least one delta carries ``reasoning``; concatenated
        reasoning is non-empty; markers never leak into either field."""
        with httpx.stream(
            "POST",
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": "chat-transformers-reasoning",
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

    def test_reasoning_with_tools_transformers(self, client):
        """Reasoning + tool calling in one round-trip through transformers.

        Mirrors ``TestChatLlamaCppReasoning.test_reasoning_with_tools_llama_cpp``:
        the single-pass ``ChatOutputStreamer`` must populate BOTH
        ``message.reasoning`` and ``message.tool_calls``.
        """
        completion = client.chat.completions.create(
            model="chat-transformers-reasoning",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
        )
        message = completion.choices[0].message
        reasoning = getattr(message, "reasoning", None) or message.model_extra.get("reasoning")
        assert reasoning, f"expected reasoning, got message={message!r}"
        assert "<think>" not in reasoning
        tool_calls = message.tool_calls
        assert tool_calls, f"expected a tool call, got content={message.content!r}, reasoning={reasoning!r}"
        assert tool_calls[0].function.name == "get_weather"
        assert "Paris" in tool_calls[0].function.arguments
        assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.integration
@pytest.mark.llama_cpp
class TestChatLimited:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-limited")

    def test_tool_calling_explicit_opt_out(self, client):
        """Verifies that ``tool_calls_enabled: false`` disables tools even when the model's chat template supports them."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        completion = client.chat.completions.create(
            model="chat-limited", messages=[{"role": "user", "content": "Weather in London?"}], tools=tools
        )
        assert not completion.choices[0].message.tool_calls


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


@pytest.mark.integration
@pytest.mark.function_gemma
class TestChatTransformersFunctionGemma:
    """End-to-end FunctionGemma (Gemma 2) tool calling through transformers.

    Verifies that the `function_gemma` parser correctly intercepts the
    `<start_function_call>` markers and `<escape>` syntax.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-transformers-function-gemma")

    def test_tool_calling_transformers_function_gemma_loader(self, client):
        completion = client.chat.completions.create(
            model="chat-transformers-function-gemma",
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

    def test_tool_calling_streaming_transformers_function_gemma_loader(self, client):
        stream = client.chat.completions.create(
            model="chat-transformers-function-gemma",
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
        assert call_0["id"], "expected an id on the first tool-call delta"
        assert call_0["name"] == "get_weather"
        assert collected["name_deltas"] == 1, f"expected one name delta, got {collected['name_deltas']}"
        # HF's ``TextIteratorStreamer`` only emits up to the last whitespace
        # character (transformers/generation/streamers.py: ``text.rfind(" ")+1``),
        # and FunctionGemma's args body contains no internal spaces, so the
        # whole body arrives in a single chunk and produces exactly one args
        # delta. vLLM / llama_cpp loaders emit per-token and still satisfy >= 2.
        assert collected["args_deltas"] >= 1, f"expected at least one args delta, got {collected['args_deltas']}"
        parsed_args = json.loads(call_0["arguments"])
        assert parsed_args.get("city")
        assert "Paris" in parsed_args["city"]
        assert collected["finish_reason"] == "tool_calls"


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
    """End-to-end /v1/responses through the stateless adapter over the vLLM
    chat pipeline. Verifies the official OpenAI SDK's ``responses.create``
    parses our payload and that unsupported features are rejected, not
    silently dropped."""

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
        # Phase A never persists, so the echoed store flag must be False.
        assert resp.store is False

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

    def test_previous_response_id_rejected_400(self):
        response = httpx.post(
            f"{OPENAI_API_BASE}/responses",
            json={"model": "chat-capable", "input": "hi", "previous_response_id": "resp_does_not_exist"},
            timeout=60,
        )
        assert response.status_code == 400, response.text
        assert "previous_response_id" in response.json()["error"]["message"]

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


@pytest.mark.integration
@pytest.mark.llama_cpp
class TestResponsesLlamaCpp:
    """The adapter is loader-agnostic: /v1/responses works over the llama_cpp
    chat pipeline with no loader-side changes, same as it does over vLLM."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("chat-llama-mship")

    def test_basic_response_through_llama_cpp(self, client):
        resp = client.responses.create(
            model="chat-llama-mship",
            input="Say hello in one word.",
            max_output_tokens=20,
        )
        assert resp.status in {"completed", "incomplete"}
        assert resp.output_text.strip()

    def test_streaming_response_through_llama_cpp(self, client):
        # The streaming translator is loader-agnostic too: it consumes the same
        # chat SSE chunk stream llama_cpp emits.
        stream = client.responses.create(
            model="chat-llama-mship",
            input="Say hello in one word.",
            max_output_tokens=20,
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
@pytest.mark.llama_cpp
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
@pytest.mark.llama_cpp
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
        model_deployer.deploy("chat-limited")
        assert _poll(lambda: _model_in_all_samples(client, "chat-limited"), deadline_s=60), (
            "deployed model did not become routable on all gateway replicas"
        )
        completion = client.chat.completions.create(
            model="chat-limited", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
        assert completion.choices[0].message.content is not None

        # Reconcile to a different model — chat-limited is removed everywhere.
        model_deployer.deploy("chat-llama-mship")
        assert _poll(lambda: _model_in_no_samples(client, "chat-limited"), deadline_s=60), (
            "removed model still routable on some gateway replica"
        )

        # Requests to the removed model now 404 on every replica — none route into
        # the torn-down deployment (which would surface as a 5xx, not a 404).
        import openai

        for _ in range(20):
            with pytest.raises(openai.NotFoundError):
                client.chat.completions.create(
                    model="chat-limited", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )
