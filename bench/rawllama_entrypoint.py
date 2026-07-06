"""Run `llama-server` directly using the same models.yaml modelship reads.

Mounted-only entrypoint — bypasses ray + modelship pipeline so a benchmark can
A/B the modelship llama_server loader against vanilla llama-server with an
identical launch command and the identical llama-server binary that ships in
the image (MSHIP_LLAMA_SERVER_BIN). Mirrors the flag-building logic in
modelship/infer/llama_server/llama_server_infer.py's `_launch` — see that
module for the source of truth this must stay in sync with.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from modelship.infer.infer_config import LlamaServerConfig, ModelLoader, ModelshipConfig, ModelUsecase
from modelship.infer.model_resolver import resolve_model_source

CONFIG_PATH = Path(os.environ.get("MSHIP_CONFIG", "/modelship/config/models.yaml"))


def main() -> int:
    binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
    if not binary or not os.path.isfile(binary):
        print(f"MSHIP_LLAMA_SERVER_BIN must point at a llama-server executable; got {binary!r}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = ModelshipConfig.model_validate(raw)
    llama_models = [m for m in cfg.models if m.loader == ModelLoader.llama_server]
    if len(llama_models) != 1:
        print(
            f"bench expects exactly one llama_server model in {CONFIG_PATH}, got {len(llama_models)}", file=sys.stderr
        )
        return 2

    m = llama_models[0]
    k = m.llama_server_config or LlamaServerConfig()

    model_path = resolve_model_source(m.model)
    print(f"rawllama resolved model -> {model_path}", flush=True)

    args = [
        binary,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "-m",
        model_path,
        "-c",
        str(k.n_ctx * k.parallel),
        "-b",
        str(k.n_batch),
        "--parallel",
        str(k.parallel),
        "--jinja",
        "--reasoning-format",
        "auto",
        "--no-webui",
        # Reported as the "id" in /v1/models — lets the harness's wait_ready
        # (which greps for the served name) work identically to the modelship
        # phase. Vanilla llama-server has no auth by default; skipping
        # --api-key here matches an out-of-the-box `llama-server` invocation.
        "--alias",
        m.name,
    ]
    # Same branch as LlamaServerInfer._launch: Ray only sets CUDA_VISIBLE_DEVICES
    # for actors that reserve GPUs, so a num_gpus=0 deploy may still see every
    # GPU — force no offload rather than trusting the container's device
    # visibility. Mirrored here so the raw phase can't accidentally offload
    # when the modelship phase wouldn't.
    if m.num_gpus > 0:
        args += ["-ngl", str(k.n_gpu_layers)]
    else:
        args += ["-ngl", "0"]
    if k.threads is not None:
        args += ["--threads", str(k.threads)]
    if k.chat_template:
        flag = "--chat-template-file" if os.path.isfile(k.chat_template) else "--chat-template"
        args += [flag, k.chat_template]
    if k.mmproj:
        mmproj_path = resolve_model_source(k.mmproj)
        args += ["--mmproj", mmproj_path]
    if m.usecase == ModelUsecase.embed:
        args += ["--embedding"]
    args += list(k.extra_args)

    print("rawllama exec:", " ".join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
