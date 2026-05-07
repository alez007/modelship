import json
import subprocess
import time

import httpx
import pytest
import yaml

from openai import OpenAI

OPENAI_API_BASE = "http://localhost:8000/v1"

EXPECTED_MODELS = {
    "chat-capable",
    "chat-limited",
    "chat-llama-mship",
    "chat-transformers",
    "embed-model",
    "stt-model",
    "tts-model",
}


@pytest.fixture(scope="session")
def mship_cluster(tmp_path_factory):
    """Starts a Ray cluster and mship_deploy in the background and waits for it to be ready."""
    tmp_dir = tmp_path_factory.mktemp("mship_integration")
    config_path = tmp_dir / "integration-models.yaml"
    log_path = tmp_dir / "mship_deploy.log"

    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--disable-usage-stats"], check=True)

    config = {
        "models": [
            {
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
            {
                "name": "chat-limited",
                "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
                "usecase": "generate",
                "loader": "llama_cpp",
                "num_cpus": 1,
                "llama_cpp_config": {
                    "tool_calls_enabled": False,
                },
            },
            {
                "name": "chat-llama-mship",
                "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
                "usecase": "generate",
                "loader": "llama_cpp",
                "num_cpus": 1,
            },
            {
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
            {
                "name": "embed-model",
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "usecase": "embed",
                "loader": "transformers",
                "num_cpus": 1,
            },
            {
                "name": "stt-model",
                "model": "openai/whisper-tiny",
                "usecase": "transcription",
                "loader": "transformers",
                "num_cpus": 1,
            },
            {
                "name": "tts-model",
                "model": "hexgrad/Kokoro-82M",
                "usecase": "tts",
                "loader": "custom",
                "plugin": "kokoroonnx",
                "num_cpus": 1,
                "plugin_config": {"onnx_provider": "CPUExecutionProvider"},
            },
        ]
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime, closed in cleanup
    proc = subprocess.Popen(
        ["uv", "run", "mship_deploy.py", "--config", str(config_path), "--redeploy"],
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
        start_time = time.time()
        timeout = 900  # 15 minutes (vLLM can be slow to init)
        ready = False
        seen_models: set[str] = set()

        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                break
            try:
                resp = httpx.get(f"{OPENAI_API_BASE}/models")
                if resp.status_code == 200:
                    seen_models = {m["id"] for m in resp.json().get("data", [])}
                    if EXPECTED_MODELS.issubset(seen_models):
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(15)

        if not ready:
            missing = EXPECTED_MODELS - seen_models
            log_tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
            cleanup()
            pytest.fail(
                f"Deployment failed to become ready within timeout.\n"
                f"Missing models: {sorted(missing)}\n"
                f"Seen models: {sorted(seen_models)}\n"
                f"Log file: {log_path}\n"
                f"Last 4KB of mship_deploy log:\n{log_tail}"
            )

        yield proc
    finally:
        cleanup()


@pytest.fixture(scope="session")
def client(mship_cluster):
    return OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed")


@pytest.mark.integration
def test_list_models(client):
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert "chat-capable" in model_ids
    assert "chat-limited" in model_ids
    assert "chat-transformers" in model_ids
    assert "embed-model" in model_ids
    assert "stt-model" in model_ids
    assert "tts-model" in model_ids


@pytest.mark.integration
def test_chat_completion(client):
    completion = client.chat.completions.create(
        model="chat-capable", messages=[{"role": "user", "content": "Hello!"}], max_tokens=10
    )
    assert completion.choices[0].message.content
    assert completion.model == "chat-capable"


@pytest.mark.integration
def test_chat_streaming(client):
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


@pytest.mark.integration
def test_tool_calling_success(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    completion = client.chat.completions.create(
        model="chat-capable",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="required",
    )
    assert completion.choices[0].message.tool_calls
    assert completion.choices[0].message.tool_calls[0].function.name == "get_weather"


@pytest.mark.integration
def test_tool_calling_transformers_loader(client):
    """Round-trip a Hermes-style tool call through the transformers loader.

    Uses the same Qwen2.5-0.5B-Instruct weights as the vLLM `chat-capable`
    deployment but goes through the modelship-side tool-calling toolkit
    (apply_chat_template(tools=...) on input, hermes parser on output).
    """
    tools = [
        {
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
    ]
    completion = client.chat.completions.create(
        model="chat-transformers",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
    )
    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
    assert tool_calls[0].function.name == "get_weather"
    assert "Paris" in tool_calls[0].function.arguments
    assert completion.choices[0].finish_reason == "tool_calls"


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


@pytest.mark.integration
def test_tool_calling_streaming_transformers_loader(client):
    """Stream a tool call through the transformers loader and verify the
    delta sequence matches the OpenAI streaming contract.

    Asserts:
    - the function name arrives in exactly one delta;
    - arguments arrive across multiple deltas (incremental, not buffered);
    - concatenated arguments form valid JSON containing the expected key;
    - the final delta carries ``finish_reason="tool_calls"``.
    """
    tools = [
        {
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
    ]
    stream = client.chat.completions.create(
        model="chat-transformers",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
        stream=True,
    )

    collected = _collect_streaming_tool_call(stream)

    assert collected["tool_calls"], f"expected at least one streamed tool call; got content={collected['content']!r}"
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
def test_tool_calling_streaming_vllm_loader(client):
    """Smoke-test that vLLM streaming + tool calling still works through
    the gateway. vLLM emits its own per-token deltas; we only verify that
    the gateway forwards them and that the final assistant message rebuilds
    correctly.
    """
    tools = [
        {
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
    ]
    stream = client.chat.completions.create(
        model="chat-capable",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
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
def test_tool_calling_explicit_opt_out(client):
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
def test_tool_calling_llama_cpp_loader(client):
    """Round-trip a Hermes-style tool call through the llama_cpp loader.

    Same Qwen2.5-0.5B-Instruct weights as `chat-capable` (vLLM) and
    `chat-transformers`, but in GGUF form via llama-cpp-python. Auto-detected
    hermes parser renders the prompt with `tools=...` and parses the
    `<tool_call>{...}</tool_call>` markers out of raw completion output.
    """
    tools = [
        {
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
    ]
    completion = client.chat.completions.create(
        model="chat-llama-mship",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
    )
    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls, f"expected a tool call, got content={completion.choices[0].message.content!r}"
    assert tool_calls[0].function.name == "get_weather"
    assert "Paris" in tool_calls[0].function.arguments
    assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.integration
def test_tool_calling_streaming_llama_cpp_loader(client):
    """Stream a tool call through the llama_cpp loader and verify the
    delta sequence matches the OpenAI streaming contract.

    Same shape as `test_tool_calling_streaming_transformers_loader` —
    asserts a single name delta, multiple incremental argument deltas,
    valid JSON on concatenation, and final ``finish_reason="tool_calls"``.
    """
    tools = [
        {
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
    ]
    stream = client.chat.completions.create(
        model="chat-llama-mship",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
        stream=True,
    )

    collected = _collect_streaming_tool_call(stream)

    assert collected["tool_calls"], f"expected at least one streamed tool call; got content={collected['content']!r}"
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
def test_embeddings(client):
    response = client.embeddings.create(model="embed-model", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


@pytest.mark.integration
def test_audio_speech(client):
    response = client.audio.speech.create(model="tts-model", voice="af_bella", input="Hello from integration test")
    # response.content is the binary audio data
    assert len(response.content) > 1000


@pytest.mark.integration
def test_audio_transcription(client, tmp_path):
    # Generate audio first using TTS
    audio_data = client.audio.speech.create(
        model="tts-model", voice="af_bella", input="This is a test transcription."
    ).content

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(audio_data)

    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(model="stt-model", file=f)
    assert "test" in transcription.text.lower()
