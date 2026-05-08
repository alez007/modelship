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
