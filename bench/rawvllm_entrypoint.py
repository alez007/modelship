"""Run `vllm serve` directly using the same models.yaml modelship reads.

Mounted-only entrypoint — bypasses ray + modelship pipeline so a benchmark can
A/B the modelship loader against raw vLLM with identical engine kwargs and the
identical vLLM wheel that ships in the image.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from modelship.infer.infer_config import ModelLoader, ModelshipConfig

CONFIG_PATH = Path(os.environ.get("MSHIP_CONFIG", "/modelship/config/models.yaml"))


def main() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = ModelshipConfig.model_validate(raw)
    vllm_models = [m for m in cfg.models if m.loader == ModelLoader.vllm]
    if len(vllm_models) != 1:
        print(f"bench expects exactly one vllm model in {CONFIG_PATH}, got {len(vllm_models)}", file=sys.stderr)
        return 2

    m = vllm_models[0]
    k = m.vllm_engine_kwargs
    args = ["vllm", "serve", m.model, "--host", "0.0.0.0", "--port", "8000", "--served-model-name", m.name]
    args += ["--tensor-parallel-size", str(k.tensor_parallel_size)]
    args += ["--pipeline-parallel-size", str(k.pipeline_parallel_size)]
    args += ["--dtype", k.dtype]
    from modelship.infer.infer_config import default_gpu_memory_utilization

    gmu = k.gpu_memory_utilization
    if gmu is None:
        gmu = default_gpu_memory_utilization(m)
    args += ["--gpu-memory-utilization", str(gmu)]
    if k.max_model_len is not None:
        args += ["--max-model-len", str(k.max_model_len)]
    if k.tokenizer:
        args += ["--tokenizer", k.tokenizer]
    if k.trust_remote_code:
        args += ["--trust-remote-code"]
    if k.kv_cache_dtype:
        args += ["--kv-cache-dtype", k.kv_cache_dtype]
    if k.quantization:
        args += ["--quantization", k.quantization]
    if k.enable_auto_tool_choice:
        args += ["--enable-auto-tool-choice"]
    if k.tool_call_parser:
        args += ["--tool-call-parser", k.tool_call_parser]
    if k.enforce_eager:
        args += ["--enforce-eager"]
    # Note: modelship derives distributed_executor_backend internally ("ray" when
    # tensor_parallel_size * pipeline_parallel_size > 1, else vLLM's default).
    # It is not a config field, so the raw phase here uses vLLM's standalone
    # default executor (mp) — a fair single-node match for the bench's tp=pp=1.

    print("rawvllm exec:", " ".join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
