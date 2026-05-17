import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest
import yaml

from openai import OpenAI

OPENAI_API_BASE = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"

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
        ["uv", "run", "mship_deploy.py", "--config", str(empty_config)],
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
