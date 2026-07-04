# Plugin Development

Plugins are Python packages that extend Modelship with custom inference backends. Each plugin is a self-contained uv workspace package with its own dependencies, installed on demand.

Plugins can implement any usecase — TTS, STT, chat, embeddings, translation, image generation — not just speech synthesis.

## How plugins work

When `loader: custom` is set in `models.yaml`, Modelship imports the module named by `plugin` and expects it to expose a `ModelPlugin` class extending `BasePlugin`.

A plugin overrides only the `create_*` method(s) matching its `usecase`:

| Usecase | Method to override | Raw return type |
|---|---|---|
| `tts` | `create_speech` | `RawSpeechResponse` or `AsyncGenerator[(bytes, int), None]` |
| `transcription` | `create_transcription` | `RawTranscription` |
| `translation` | `create_translation` | `RawTranslation` |
| `generate` | `create_chat_completion` | `RawChatCompletion` or `AsyncGenerator[RawChatDelta, None]` |
| `embed` | `create_embedding` | `list[list[float]]` |
| `image` | `create_image_generation` | `list[bytes]` (PNG-encoded) |

Plugins return protocol-agnostic raw outputs. The serving wrappers in `modelship/infer/custom/openai/` translate these into OpenAI-compatible responses, so a different protocol adapter (e.g. Anthropic, gRPC) could be added later without touching any plugin.

Unimplemented methods fall back to a 404 "plugin does not support this action" error.

## Creating a plugin

### 1. Create the package structure

```
plugins/
  myplugin/
    pyproject.toml
    myplugin/
      __init__.py
      myplugin.py
```

### 2. Write `pyproject.toml`

```toml
[project]
name = "myplugin"
version = "0.1.0"
requires-python = "==3.12.10"
dependencies = [
    # only packages unique to your plugin — see "Dependency contract" below
]

[build-system]
requires = ["uv_build"]
build-backend = "uv_build"

[tool.uv.sources]
modelship = { workspace = true }

[tool.uv.build-backend]
module-name = "myplugin"
module-root = ""
```

#### Dependency contract

Plugins assume the host environment provides `modelship` itself plus the full core/gpu/cpu stack: `torch`, `torchvision`, `transformers`, `numpy`, `scipy`, `librosa`, `soundfile`, `onnxruntime[-gpu]`, `diffusers`, `vllm`. **Do not redeclare any of these in your plugin's `dependencies`.**

Why: plugin wheels are shipped to Ray workers via `runtime_env`, which installs them into a per-job venv layered over a base image that already has the core stack baked in. Redeclaring `modelship` or `torch` causes pip to either pull a second copy from PyPI (version drift, two packages on `sys.path`) or error on the layered install.

Declare only what's unique to your plugin — e.g. `kokoro-onnx`, `pywhispercpp`, `snac`. Dev-time imports of `modelship.*` resolve because the workspace's shared `.venv` always has the root `modelship` package installed; you don't need to declare it to get IDE/pyright support.

### 3. Implement `ModelPlugin`

#### TTS example

```python
# plugins/myplugin/myplugin/myplugin.py
from collections.abc import AsyncGenerator

from modelship.plugins.base_plugin import BasePlugin
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_name = model_config.model
        self.config = model_config.plugin_config or {}

    async def start(self):
        # load your model here
        pass

    async def create_speech(
        self,
        input: str,
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        request_id: str | None = None,
    ) -> RawSpeechResponse | AsyncGenerator[tuple[bytes, int], None] | ErrorResponse:
        audio_bytes = b"..."  # your synthesis here
        return RawSpeechResponse(audio=audio_bytes)
```

#### STT example

```python
from modelship.plugins.base_plugin import BasePlugin
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.openai.protocol import ErrorResponse, RawTranscription


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_name = model_config.model

    async def start(self):
        pass

    async def create_transcription(
        self,
        audio_data: bytes,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> RawTranscription | ErrorResponse:
        text = "..."  # your transcription here
        return RawTranscription(text=text, language=language, duration_seconds=0.0)
```

### 4. Export `ModelPlugin` from `__init__.py`

```python
# plugins/myplugin/myplugin/__init__.py
from myplugin.myplugin import ModelPlugin

__all__ = ["ModelPlugin"]
```

### 5. Register the extra in the root `pyproject.toml`

```toml
[project.optional-dependencies]
myplugin = ["myplugin"]

[tool.uv.sources]
myplugin = { workspace = true }
```

### 6. Install and configure

```bash
uv sync --extra myplugin
```

In `models.yaml`:

```yaml
- name: myplugin
  usecase: tts        # or transcription, translation, generate, embed, image
  loader: custom
  plugin: myplugin
  num_gpus: 0.1
```

## SSE streaming (TTS)

For streaming speech, yield `(pcm_bytes, sample_rate)` tuples from an async generator. `pcm_bytes` must be signed 16-bit little-endian mono PCM — the serving wrapper base64-encodes each chunk into SSE `speech.audio.delta` events.

```python
async def create_speech(self, input, voice=None, speed=None, stream=False, request_id=None):
    if stream:
        return self._stream(input, voice, speed)
    # non-stream path
    audio = self._synthesize_full(input)
    return RawSpeechResponse(audio=audio)

async def _stream(self, input, voice, speed):
    for pcm_chunk, sample_rate in self._synthesize_chunks(input):
        yield pcm_chunk, sample_rate
```

## Plugin README

Every plugin must include a `README.md` in its package root (`plugins/myplugin/README.md`). This is the primary documentation for users configuring the plugin. It should cover:

- **Installation** — how to install the plugin (`uv sync --extra` for local development; automatic via wheels for deployment)
- **Configuration** — example `models.yaml` entry with all `plugin_config` options documented in a table
- **Voices / options** — any model-specific choices (voice presets, providers, etc.)
- **Example request** — a working `curl` command

See the built-in plugins for reference: [Kokoro ONNX](../plugins/kokoroonnx/README.md), [Orpheus](../plugins/orpheus/README.md), [whisper.cpp](../plugins/whispercpp/README.md).

## Submitting to this repo

Open a PR adding:
- `plugins/myplugin/` with your package
- `plugins/myplugin/README.md` documenting configuration and usage
- One line in root `pyproject.toml` optional extras
