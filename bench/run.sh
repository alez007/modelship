#!/usr/bin/env bash
# A/B benchmark: a modelship loader vs the vanilla inference server it wraps,
# same image, same engine config. Answers "how much does modelship's wrapping
# cost?" — not a vllm-vs-llama.cpp comparison, so --loader picks which wrapped
# stack to measure, and --device picks CPU vs GPU for that stack.
#
# Usage: bench/run.sh [--loader vllm|llama_server] [--device gpu|cpu] [--image TAG]
#                      [--config PATH] [--num-prompts N] [--concurrency N]
#                      [--input-len N] [--output-len N] [--num-warmups N] [--repeats N]
set -euo pipefail

LOADER="vllm"
DEVICE="gpu"
IMAGE=""
CONFIG=""
TOKENIZER=""
NUM_PROMPTS=100
CONCURRENCY=8
INPUT_LEN=128
OUTPUT_LEN=512
# Warmup requests sent (and discarded) before timing begins. Warms CUDA graph
# capture / lazy compilation (GPU) or first-request JIT paths so the first few
# real requests don't eat a cold-start tail that skews mean/p99 TTFT and throughput.
NUM_WARMUPS=20
# Timed sweeps per stack. We report the median across repeats so a single cold
# or noisy run can't dominate; tail metrics on one ~100-prompt run are unstable.
REPEATS=3
READY_TIMEOUT=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loader) LOADER="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --num-warmups) NUM_WARMUPS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$LOADER" in
    vllm|llama_server) ;;
    *) echo "--loader must be vllm or llama_server, got: $LOADER" >&2; exit 2 ;;
esac
case "$DEVICE" in
    gpu|cpu) ;;
    *) echo "--device must be gpu or cpu, got: $DEVICE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/bench"
# shellcheck source=bench/lib.sh
source "$BENCH_DIR/lib.sh"

# modelship's own CLAUDE.md/Dockerfile convention: gpu/cpu are separate image
# variants (mutually exclusive extras, different vllm wheel index), and the
# published image tags a cpu build with a "-cpu" suffix.
if [[ -z "$IMAGE" ]]; then
    IMAGE="modelship:dev"
    [[ "$DEVICE" == "cpu" ]] && IMAGE="modelship:dev-cpu"
fi

CONFIG_PREFIX="vllm"
BASELINE_ENTRYPOINT="rawvllm_entrypoint.py"
BASELINE_LABEL="raw vllm"
if [[ "$LOADER" == "llama_server" ]]; then
    CONFIG_PREFIX="llama"
    BASELINE_ENTRYPOINT="rawllama_entrypoint.py"
    BASELINE_LABEL="vanilla llama-server"
fi
[[ -n "$CONFIG" ]] || CONFIG="$BENCH_DIR/configs/${CONFIG_PREFIX}-${DEVICE}.yaml"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$BENCH_DIR/results/$TS"
mkdir -p "$RESULTS_DIR"

CACHE_DIR="${MSHIP_CACHE_DIR:-$REPO_ROOT/models-cache}"
mkdir -p "$CACHE_DIR"

SERVED_NAME="$(yaml_scalar '^[[:space:]]*-[[:space:]]*name:' "$CONFIG")"
MODEL_ID="$(yaml_scalar '^[[:space:]]*model:' "$CONFIG")"
[[ -n "$MODEL_ID" && -n "$SERVED_NAME" ]] || { echo "failed to parse $CONFIG" >&2; exit 2; }

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$(yaml_scalar '^[[:space:]]*#[[:space:]]*bench-tokenizer:' "$CONFIG")"
fi
if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$MODEL_ID"
fi

NUM_CPUS="$(yaml_scalar '^[[:space:]]*num_cpus:' "$CONFIG")"
BASELINE_ENV_ARGS=()
if [[ -n "${NUM_CPUS:-}" ]]; then
    BASELINE_ENV_ARGS+=(-e "OMP_NUM_THREADS=$NUM_CPUS")
fi

MODELSHIP_CONTAINER=bench-modelship
BASELINE_CONTAINER=bench-baseline
trap cleanup EXIT

# Defensive: remove any pre-existing bench containers from a prior aborted run.
docker rm -f "$MODELSHIP_CONTAINER" "$BASELINE_CONTAINER" >/dev/null 2>&1 || true

DOCKER_GPU_ARGS=()
[[ "$DEVICE" == "gpu" ]] && DOCKER_GPU_ARGS=(--gpus all)

start_modelship() {
    # Mount local source over the prebuilt image so the bench exercises the
    # working tree. mship_deploy.py is the entry point invoked by the image's
    # default CMD (scripts/start.sh).
    #
    # MSHIP_PREFLIGHT=false ensures modelship loader falls back to standard defaults
    # matching the baseline phase exactly.
    docker run -d "${DOCKER_GPU_ARGS[@]}" --ipc=host --network host \
        -e MSHIP_METRICS=true \
        -e MSHIP_PREFLIGHT=false \
        -e MSHIP_GATEWAY_REPLICAS="${MSHIP_GATEWAY_REPLICAS:-1}" \
        -e MSHIP_GATEWAY_MAX_ONGOING="${MSHIP_GATEWAY_MAX_ONGOING:-1024}" \
        -v "$CONFIG:/modelship/config/models.yaml:ro" \
        -v "$REPO_ROOT/mship_deploy.py:/modelship/mship_deploy.py:ro" \
        -v "$REPO_ROOT/modelship:/modelship/modelship:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        --name "$MODELSHIP_CONTAINER" "$IMAGE" >/dev/null
}

start_baseline() {
    # Mount the local modelship code into the baseline container so that
    # any pydantic schema or other changes in the working tree are shared.
    docker run -d "${DOCKER_GPU_ARGS[@]}" --ipc=host --network host \
        -e PYTHONPATH=/modelship \
        "${BASELINE_ENV_ARGS[@]}" \
        -v "$CONFIG:/modelship/config/models.yaml:ro" \
        -v "$REPO_ROOT/modelship:/modelship/modelship:ro" \
        -v "$BENCH_DIR/$BASELINE_ENTRYPOINT:/modelship/bench/$BASELINE_ENTRYPOINT:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        -w /modelship \
        --entrypoint /.venv/bin/python \
        --name "$BASELINE_CONTAINER" "$IMAGE" \
        "/modelship/bench/$BASELINE_ENTRYPOINT" >/dev/null
}

echo "=== bench $TS — loader=$LOADER device=$DEVICE image=$IMAGE config=$(basename "$CONFIG") prompts=$NUM_PROMPTS conc=$CONCURRENCY in=$INPUT_LEN out=$OUTPUT_LEN warmups=$NUM_WARMUPS repeats=$REPEATS ==="

# Phase A — modelship
echo "[A] starting modelship..."
start_modelship
wait_ready "$MODELSHIP_CONTAINER"
echo "[A] running $REPEATS sweep(s)..."
warm_model_cache
mkdir -p "$RESULTS_DIR/modelship"
start_mem_sampler modelship "$MODELSHIP_CONTAINER"
start_component_sampler "$RESULTS_DIR/modelship/components.txt"
run_stack modelship
stop_mem_sampler
stop_component_sampler
scrape_prom "$RESULTS_DIR/modelship/prom.txt"
docker logs "$MODELSHIP_CONTAINER" > "$RESULTS_DIR/${MODELSHIP_CONTAINER}.log" 2>&1 || true
docker rm -f "$MODELSHIP_CONTAINER" >/dev/null
vram_gate

# Phase B — baseline (vanilla vllm or vanilla llama-server, same image/config)
echo "[B] starting baseline ($BASELINE_LABEL)..."
start_baseline
wait_ready "$BASELINE_CONTAINER"
echo "[B] running $REPEATS sweep(s)..."
warm_model_cache
mkdir -p "$RESULTS_DIR/baseline"
start_mem_sampler baseline "$BASELINE_CONTAINER"
run_stack baseline
stop_mem_sampler
docker logs "$BASELINE_CONTAINER" > "$RESULTS_DIR/${BASELINE_CONTAINER}.log" 2>&1 || true
docker rm -f "$BASELINE_CONTAINER" >/dev/null

# Launch parity (config correctness): fail before summarizing if the two arms
# weren't launched with identical engine args — nothing downstream is meaningful.
assert_launch_parity

# Summary
SUMMARY="$RESULTS_DIR/summary.md"
{
    echo "# bench $TS — $LOADER / $DEVICE"
    echo
    echo "image: \`$IMAGE\`  config: \`$(basename "$CONFIG")\`  prompts: $NUM_PROMPTS  concurrency: $CONCURRENCY  input/output: $INPUT_LEN/$OUTPUT_LEN  warmups: $NUM_WARMUPS  repeats: $REPEATS"
    echo
    echo "Values are the median across \`repeats\` sweeps."
    echo
    write_summary modelship baseline "$BASELINE_LABEL"
} | tee "$SUMMARY"

echo
echo "results: $RESULTS_DIR"

# Result-population gate LAST: the summary above is written first (so the
# completed/failed rows and token-parity line are always available for
# inspection), then we fail the run with a non-zero exit if either arm dropped
# requests — so a survivorship-biased comparison is never treated as a pass.
assert_result_parity
