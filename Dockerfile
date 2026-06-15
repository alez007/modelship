ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10
ARG MSHIP_VARIANT=gpu
ARG UID=1000
ARG GID=1000

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
        gnupg \
        gosu \
        libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Register the NVIDIA CUDA apt repo and install cuda-cudart (GPU variant only).
# gcc + libc6-dev stay because torch/triton JIT-compile kernels at model-load
# time and shell out to $CC; without them, vllm crashes in _inductor with
# "Failed to find C compiler".
RUN if [ "$MSHIP_VARIANT" = "gpu" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends cuda-cudart-${CUDA_VERSION_DASH} && \
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

RUN if [ "$MSHIP_VARIANT" = "gpu" ]; then \
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
    if [ "$MSHIP_VARIANT" = "gpu" ]; then \
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

ENTRYPOINT ["/modelship/scripts/entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "mship_deploy.py"]
