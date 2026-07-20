ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10
ARG MSHIP_VARIANT=cuda
ARG UID=1000
ARG GID=1000

# llama.cpp's official server images, pinned by manifest digest (the digest IS
# the sha256 of the content). Upstream publishes no Linux CUDA binary in its
# GitHub releases, so the CUDA variant sources their CUDA 13 Docker build
# instead. Bump the tag and digest together.
ARG LLAMA_CPP_IMAGE_CUDA=ghcr.io/ggml-org/llama.cpp:server-cuda13-b9859@sha256:e8e003c66cb77615dfef2f6ae1b7f5ad0de7bd048e19c40357220cb4141d1cdc
ARG LLAMA_CPP_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b9859@sha256:f415de2e2c3e61b3dfab40d7fd26136c13d342c1ae4b3ffa8657fcc6a2f43d60

# =============================================================================
# llama-server — assembles /opt/llama.cpp for the llama_server loader.
#
# /app in the upstream images holds the llama-server binary plus the .so
# backends it dlopen()s (GGML_BACKEND_DL). The CUDA backend is skipped
# gracefully when no GPU/driver is present, so the CUDA build also runs
# CPU-only. libggml-cuda.so dynamically links libcudart/libcublas(Lt) — copied
# in from the same image so they can't skew against the venv's torch-bundled
# CUDA libs — and the driver's libcuda.so.1, which the NVIDIA Container
# Toolkit provides at run time.
# =============================================================================
FROM ${LLAMA_CPP_IMAGE_CUDA} AS llama-server-cuda

RUN set -e && \
    mkdir -p /opt/llama.cpp && \
    cp -a /app/. /opt/llama.cpp/ && \
    ldconfig && \
    for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do \
        cp -L "$(ldconfig -p | awk -v lib="$lib" '$1 == lib { print $NF; exit }')" /opt/llama.cpp/; \
    done

FROM ${LLAMA_CPP_IMAGE_CPU} AS llama-server-cpu

RUN mkdir -p /opt/llama.cpp && cp -a /app/. /opt/llama.cpp/

# thin ships no llama-server binary — it never loads a model. The wrapper
# script + MSHIP_LLAMA_SERVER_BIN below still get written pointing at this
# empty dir; harmless, since thin never invokes it.
FROM ubuntu:24.04 AS llama-server-thin

RUN mkdir -p /opt/llama.cpp

# The raw binary can't run standalone: it resolves its sibling .so files via
# the loader path. The wrapper scopes LD_LIBRARY_PATH to the llama-server
# process only — globally, /opt/llama.cpp would shadow llama-cpp-python's own
# bundled libllama/libggml.
FROM llama-server-${MSHIP_VARIANT} AS llama-server

RUN printf '#!/bin/sh\nexport LD_LIBRARY_PATH="/opt/llama.cpp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec /opt/llama.cpp/llama-server "$@"\n' \
        > /opt/llama.cpp/llama-server.sh && \
    chmod +x /opt/llama.cpp/llama-server.sh

# =============================================================================
# base — minimal runtime OS + uv + non-root user + env vars.
#
# CUDA strategy: torch cu130 bundles libcublas/libcudnn/libcurand/libnccl/
# libnvrtc inside the venv (under site-packages/nvidia/*/lib) and the NVIDIA
# Container Toolkit provides libcuda.so at run time via --gpus. However, vLLM's
# C extensions (_C.abi3.so, _moe_C.abi3.so, ...) are built with an RPATH that
# hard-references /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13.
# Without that file, the vLLM registry subprocess that runs before torch has
# bootstrapped its dlopen paths crashes with malloc_consolidate/SIGABRT while
# the dynamic loader resolves symbols. We therefore install ONLY the tiny
# cuda-cudart runtime package (~800 KB) in the base image. libcublas/cudnn/
# curand/nvrtc are NOT installed — torch's bundled copies are resolved via its
# own rpath once Python imports torch, so adding system duplicates is wasted
# space.
# =============================================================================
FROM ubuntu:24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        espeak-ng \
        gcc \
        g++ \
        gnupg \
        gosu \
        libc6-dev \
        libgomp1 \
        libnuma1 \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Register the NVIDIA CUDA apt repo and install cuda-cudart, cuda-nvcc, and
# cuda-cuobjdump (cuda variant only). gcc/g++ + libc6-dev and ninja-build stay
# because torch/triton and flashinfer JIT-compile kernels at model-load time
# and shell out to $CC/nvcc; without them, vllm crashes in _inductor (needs
# g++ for its CPU codegen backend, e.g. the vllm CPU loader's torch.compile
# path) or flashinfer on newer architectures (such as Blackwell).
RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} && \
    apt-get purge -y --auto-remove gnupg && \
    rm -f /etc/apt/sources.list.d/cuda.list /usr/share/keyrings/cuda-keyring.gpg && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN if ! getent group $GID >/dev/null; then groupadd -g $GID modelship; fi && \
    if ! getent passwd $UID >/dev/null; then useradd -m -u $UID -g $GID modelship; \
    else existing=$(getent passwd $UID | cut -d: -f1) && usermod -l modelship -d /home/modelship -m "$existing"; fi

ENV MSHIP_UID=$UID
ENV MSHIP_GID=$GID

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /modelship

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV MSHIP_CACHE_DIR=/.cache
ENV UV_CACHE_DIR=${MSHIP_CACHE_DIR}/uv
ENV MSHIP_USE_EXISTING_RAY_CLUSTER=false
ENV MSHIP_METRICS=true
ENV RAY_METRICS_EXPORT_PORT=8079
ENV MSHIP_LOG_LEVEL=INFO
ENV MSHIP_LOG_FORMAT=text
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv/python
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV MSHIP_PLUGIN_WHEEL_DIR=/opt/modelship/plugin-wheels

# Pinned llama.cpp build for the llama_server loader (see the llama-server
# stages above). llama-server additionally needs libgomp1 (ggml CPU backends)
# and libssl3 (already pulled in via curl).
COPY --from=llama-server /opt/llama.cpp /opt/llama.cpp
ENV MSHIP_LLAMA_SERVER_BIN=/opt/llama.cpp/llama-server.sh

# onnxruntime-gpu (pulled in by the kokoroonnx plugin) dlopen()s
# libonnxruntime_providers_cuda.so which has plain DT_NEEDED entries for
# libcublasLt.so.13 / libcudnn.so.9 / etc. Torch cu130 bundles these under
# site-packages/nvidia/*/lib and resolves them via its own rpath once imported
# — but onnxruntime doesn't participate in that. Expose the torch-bundled
# CUDA libs on LD_LIBRARY_PATH so onnxruntime's CUDA provider can load.
# Python version is pinned via PYTHON_VERSION (see pyproject.toml); we hard-
# code 3.12 here because UV_PROJECT_ENVIRONMENT is fixed and the ENV cannot
# shell-evaluate.
ENV LD_LIBRARY_PATH="/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib"

RUN mkdir -p /.cache /.venv $MSHIP_PLUGIN_WHEEL_DIR /usr/local/uv/python && \
    chown -R $UID:$GID /modelship /.cache /.venv $MSHIP_PLUGIN_WHEEL_DIR /usr/local/uv/python

# =============================================================================
# builder — adds build toolchain (nvcc, build-essential, dev headers, git) and
# re-registers the NVIDIA apt repo so we can pull nvcc / dev headers needed to
# compile wheels from source (flashinfer, llama-cpp-python, etc.). All of this
# stays in the builder stage and is NOT copied into prod.
#
# The venv is resolved with --extra $MSHIP_VARIANT only (no plugin extras).
# Plugin wheels are built separately into $MSHIP_PLUGIN_WHEEL_DIR and shipped
# to Ray workers per-deployment via runtime_env from mship_deploy.py.
# =============================================================================
FROM base AS builder

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    apt-get update -y && \
    apt-get install -y --no-install-recommends gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git && \
    if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH}; \
    fi && \
    rm -rf /var/lib/apt/lists/*

USER modelship

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv python install ${PYTHON_VERSION}
RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv venv

ADD --chown=$UID:$GID ./pyproject.toml pyproject.toml
ADD --chown=$UID:$GID ./README.md README.md
ADD --chown=$UID:$GID ./uv.lock uv.lock
ADD --chown=$UID:$GID ./Makefile Makefile
ADD --chown=$UID:$GID ./plugins plugins

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv sync --locked --no-install-project --extra $MSHIP_VARIANT

# Build plugin wheels into $MSHIP_PLUGIN_WHEEL_DIR. Plugins are NOT installed
# into /.venv — they ship to Ray workers per-deployment via runtime_env, so the
# prod venv stays lean.
RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    make plugin-wheels

# =============================================================================
# dev — inherits builder (keeps toolchain) and adds dev extras PLUS plugin
# extras so developers get editable installs for interactive REPL / pytest.
# Prod does not inherit this stage.
# =============================================================================
FROM builder AS dev

ARG MSHIP_VARIANT
ARG UID
ARG GID

USER modelship

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv sync --locked --no-install-project --extra dev --extra $MSHIP_VARIANT

USER root

ENTRYPOINT ["/modelship/scripts/entrypoint.sh"]

# =============================================================================
# prod — minimal runtime. No build tools. Copies the resolved venv and
# Python interpreter from builder.
# =============================================================================
FROM base AS prod

ARG UID
ARG GID

COPY --from=builder --chown=$UID:$GID /usr/local/uv/python /usr/local/uv/python
COPY --from=builder --chown=$UID:$GID /.venv /.venv
COPY --from=builder --chown=$UID:$GID $MSHIP_PLUGIN_WHEEL_DIR $MSHIP_PLUGIN_WHEEL_DIR

ADD --chown=$UID:$GID ./pyproject.toml pyproject.toml
ADD --chown=$UID:$GID ./README.md README.md
ADD --chown=$UID:$GID ./uv.lock uv.lock
ADD --chown=$UID:$GID ./plugins plugins
ADD --chown=$UID:$GID ./mship_deploy.py mship_deploy.py
ADD --chown=$UID:$GID ./modelship modelship
ADD --chown=$UID:$GID ./scripts scripts

USER root

ENTRYPOINT ["/modelship/scripts/entrypoint.sh", "--serve"]

# thin variant: pin capacity to 0 so it never advertises resources it can't
# serve (no torch/vllm). Layered on top since a shared stage can't
# conditionally set ENV.
FROM prod AS prod-thin

ENV MSHIP_NODE_NUM_CPUS=0
ENV MSHIP_NODE_NUM_GPUS=0
