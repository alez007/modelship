#!/usr/bin/env bash
# A/B benchmark: modelship vLLM loader vs raw vLLM, same image, same config.
# Usage: bench/run.sh [--image TAG] [--num-prompts N] [--concurrency N] [--input-len N] [--output-len N]
set -euo pipefail

IMAGE="${IMAGE:-modelship:prod}"
NUM_PROMPTS=100
CONCURRENCY=8
INPUT_LEN=128
OUTPUT_LEN=512
READY_TIMEOUT=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/bench"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$BENCH_DIR/results/$TS"
mkdir -p "$RESULTS_DIR"

CACHE_DIR="${MSHIP_CACHE_DIR:-$REPO_ROOT/models-cache}"
mkdir -p "$CACHE_DIR"

# Extract the first scalar matching a key regex from bench.yaml. Tolerates
# double-quoted, single-quoted, and unquoted values: strips everything up to
# the first colon, trims surrounding whitespace, then removes a matched pair of
# surrounding quotes (only when both ends use the same quote char).
yaml_scalar() {
    grep -m1 -E "$1" "$BENCH_DIR/configs/bench.yaml" \
        | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*\$//; s/^(['\"])(.*)\1\$/\2/"
}
SERVED_NAME="$(yaml_scalar '^[[:space:]]*-[[:space:]]*name:')"
MODEL_ID="$(yaml_scalar '^[[:space:]]*model:')"
[[ -n "$MODEL_ID" && -n "$SERVED_NAME" ]] || { echo "failed to parse bench.yaml" >&2; exit 2; }

cleanup() {
    if [[ -n "${MEM_SAMPLER_PID:-}" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill "$MEM_SAMPLER_PID" 2>/dev/null || true
        wait "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    for c in bench-modelship bench-rawvllm; do
        if docker inspect "$c" >/dev/null 2>&1; then
            docker logs "$c" >"$RESULTS_DIR/${c}.container.log" 2>&1 || true
            docker rm -f "$c" >/dev/null 2>&1 || true
        fi
    done
}
trap cleanup EXIT

# Defensive: remove any pre-existing bench containers from a prior aborted run.
docker rm -f bench-modelship bench-rawvllm >/dev/null 2>&1 || true

wait_ready() {
    local name="$1"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        # /v1/models reachable AND lists the served model id
        if curl -fsS http://localhost:8000/v1/models 2>/dev/null \
            | grep -q "\"id\":\"$SERVED_NAME\""; then
            return 0
        fi
        if ! docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q "$name"; then
            echo "container $name died" >&2
            docker logs --tail 80 "$name" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "timeout waiting for $name to be ready (served=$SERVED_NAME)" >&2
    docker logs --tail 80 "$name" >&2 || true
    return 1
}

start_mem_sampler() {
    local stack="$1"
    local container="$2"
    local out="$RESULTS_DIR/$stack/mem.tsv"
    : > "$out"
    (
        while :; do
            local ts vram cmem
            ts=$(date +%s)
            # || true: under pipefail+set -e a failing nvidia-smi/docker stats
            # would otherwise abort this backgrounded subshell and silently stop
            # sampling. Empty values fall back to 0 in the printf below.
            vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || true
            # Mem usage like "1.234GiB / 64GiB" — take first field, normalize to MiB.
            cmem=$(docker stats --no-stream --format '{{.MemUsage}}' "$container" 2>/dev/null \
                | awk -F'/' '{print $1}' \
                | awk '{
                    v=$1; u=$1;
                    sub(/[0-9.]+/,"",u); sub(/[A-Za-z]+$/,"",v);
                    if (u=="GiB") v*=1024;
                    else if (u=="MiB") v*=1;
                    else if (u=="KiB") v/=1024;
                    else if (u=="B")   v/=1048576;
                    printf "%.1f", v
                  }') || true
            printf '%s\t%s\t%s\n' "${ts:-0}" "${vram:-0}" "${cmem:-0}" >> "$out"
            sleep 1
        done
    ) &
    MEM_SAMPLER_PID=$!
}

stop_mem_sampler() {
    if [[ -n "${MEM_SAMPLER_PID:-}" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill "$MEM_SAMPLER_PID" 2>/dev/null || true
        wait "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    MEM_SAMPLER_PID=""
}

vram_gate() {
    local deadline=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < deadline )); do
        local used
        # tr -dc digits → "" when nvidia-smi is missing, errors, or prints
        # non-numeric output; || true keeps pipefail+set -e from aborting the
        # run. Guard the arithmetic so an empty operand isn't a syntax error.
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9') || true
        if [[ -n "$used" ]] && (( used < 500 )); then return 0; fi
        sleep 1
    done
    echo "warn: VRAM not freed within 60s" >&2
}

run_sweep() {
    local stack="$1"
    local out_dir="$RESULTS_DIR/$stack"
    mkdir -p "$out_dir"
    chmod 777 "$out_dir"
    docker run --rm --network host -v "$out_dir:/out:rw" "$IMAGE" \
        bash -lc "cd /modelship && uv run --active --no-sync vllm bench serve \
            --backend openai-chat \
            --base-url http://localhost:8000 \
            --endpoint /v1/chat/completions \
            --model $SERVED_NAME \
            --tokenizer $MODEL_ID \
            --dataset-name random \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $CONCURRENCY \
            --save-result \
            --result-dir /out \
            --result-filename result.json"
}

start_modelship() {
    # Mount local source over the prebuilt image so the bench exercises the
    # working tree. mship_deploy.py is the entry point invoked by the image's
    # default CMD (scripts/start.sh).
    docker run -d --gpus all --ipc=host --network host \
        -e MSHIP_METRICS=true \
        -e MSHIP_GATEWAY_REPLICAS="${MSHIP_GATEWAY_REPLICAS:-1}" \
        -e MSHIP_GATEWAY_MAX_ONGOING="${MSHIP_GATEWAY_MAX_ONGOING:-1024}" \
        -v "$BENCH_DIR/configs/bench.yaml:/modelship/config/models.yaml:ro" \
        -v "$REPO_ROOT/mship_deploy.py:/modelship/mship_deploy.py:ro" \
        -v "$REPO_ROOT/modelship:/modelship/modelship:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        --name bench-modelship "$IMAGE" >/dev/null
}

start_rawvllm() {
    docker run -d --gpus all --ipc=host --network host \
        -e PYTHONPATH=/modelship \
        -v "$BENCH_DIR/configs/bench.yaml:/modelship/config/models.yaml:ro" \
        -v "$BENCH_DIR/rawvllm_entrypoint.py:/modelship/bench/rawvllm_entrypoint.py:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        -w /modelship \
        --entrypoint /.venv/bin/python \
        --name bench-rawvllm "$IMAGE" \
        /modelship/bench/rawvllm_entrypoint.py >/dev/null
}

scrape_prom() {
    local out="$1"
    curl -fsS http://localhost:8079/metrics 2>/dev/null \
        | awk '/^ray_modelship_(request|generation)_duration_seconds_(sum|count)/ \
              || /^ray_serve_request_router_fulfillment_time_ms_(sum|count)/' \
        > "$out" || true
}

echo "=== bench $TS — image=$IMAGE prompts=$NUM_PROMPTS conc=$CONCURRENCY in=$INPUT_LEN out=$OUTPUT_LEN ==="

# Phase A — modelship
echo "[A] starting modelship..."
start_modelship
wait_ready bench-modelship
echo "[A] running sweep..."
mkdir -p "$RESULTS_DIR/modelship"
start_mem_sampler modelship bench-modelship
run_sweep modelship
stop_mem_sampler
scrape_prom "$RESULTS_DIR/modelship/prom.txt"
docker rm -f bench-modelship >/dev/null
vram_gate

# Phase B — rawvllm
echo "[B] starting rawvllm..."
start_rawvllm
wait_ready bench-rawvllm
echo "[B] running sweep..."
mkdir -p "$RESULTS_DIR/rawvllm"
start_mem_sampler rawvllm bench-rawvllm
run_sweep rawvllm
stop_mem_sampler
docker rm -f bench-rawvllm >/dev/null

# Summary
SUMMARY="$RESULTS_DIR/summary.md"
{
    echo "# bench $TS"
    echo
    echo "image: \`$IMAGE\`  prompts: $NUM_PROMPTS  concurrency: $CONCURRENCY  input/output: $INPUT_LEN/$OUTPUT_LEN"
    echo
    echo "| metric | modelship | rawvllm | overhead |"
    echo "| --- | ---: | ---: | ---: |"
    python3 - "$RESULTS_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def load(stack):
    return json.loads((root / stack / "result.json").read_text())
m = load("modelship"); r = load("rawvllm")
keys = [
    ("request_throughput", "req/s", 3),
    ("output_throughput",  "output tok/s", 2),
    ("mean_ttft_ms",       "TTFT mean (ms)", 1),
    ("p50_ttft_ms",        "TTFT p50 (ms)", 1),
    ("p95_ttft_ms",        "TTFT p95 (ms)", 1),
    ("mean_itl_ms",        "ITL mean (ms)", 2),
    ("p95_itl_ms",         "ITL p95 (ms)", 2),
    ("mean_e2el_ms",       "E2E mean (ms)", 1),
    ("p50_e2el_ms",        "E2E p50 (ms)", 1),
    ("p95_e2el_ms",        "E2E p95 (ms)", 1),
]
for key, label, prec in keys:
    mv = m.get(key); rv = r.get(key)
    if mv is None or rv is None:
        continue
    if rv == 0:
        ratio = "—"
    else:
        ratio = f"{(mv - rv) / rv * 100:+.1f}%"
    print(f"| {label} | {mv:.{prec}f} | {rv:.{prec}f} | {ratio} |")
PY
    echo
    echo "## memory (peak during sweep)"
    python3 - "$RESULTS_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
def peak(stack):
    f = root / stack / "mem.tsv"
    if not f.exists():
        return None, None
    vmax = cmax = 0.0
    for line in f.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            v = float(parts[1]); c = float(parts[2])
        except ValueError:
            continue
        if v > vmax: vmax = v
        if c > cmax: cmax = c
    return vmax, cmax
mv, mc = peak("modelship"); rv, rc = peak("rawvllm")
print("| metric | modelship | rawvllm | overhead |")
print("| --- | ---: | ---: | ---: |")
def row(label, m, r, unit):
    if m is None or r is None:
        return
    delta = m - r
    pct = f"{(delta / r * 100):+.1f}%" if r else "—"
    print(f"| {label} | {m:.0f} {unit} | {r:.0f} {unit} | {delta:+.0f} {unit} ({pct}) |")
row("peak VRAM (GPU0)", mv, rv, "MiB")
row("peak container RSS", mc, rc, "MiB")
PY
    echo
    echo "## modelship internal (Prometheus)"
    if [[ -s "$RESULTS_DIR/modelship/prom.txt" ]]; then
        python3 - "$RESULTS_DIR/modelship/prom.txt" <<'PY'
import sys, re
sums = {}; counts = {}
pat = re.compile(
    r'(ray_modelship_(?:request|generation)_duration_seconds'
    r'|ray_serve_request_router_fulfillment_time_ms)'
    r'_(sum|count)\S*\s+([0-9eE+\-.]+)'
)
for line in open(sys.argv[1]):
    m = pat.match(line)
    if not m: continue
    name, kind, val = m.group(1), m.group(2), float(m.group(3))
    (sums if kind=="sum" else counts).setdefault(name, 0.0)
    if kind == "sum": sums[name] += val
    else: counts[name] += val
def mean(n, scale=1.0):
    return (sums.get(n, 0.0) / counts[n] * scale) if counts.get(n) else float("nan")
e2e = mean("ray_modelship_request_duration_seconds")            # seconds
eng = mean("ray_modelship_generation_duration_seconds")         # seconds
qms = mean("ray_serve_request_router_fulfillment_time_ms")      # already ms
print(f"- mean E2E (gateway):       **{e2e*1000:.1f} ms**")
print(f"- mean engine (vllm):       **{eng*1000:.1f} ms**")
print(f"- mean router queue wait:   **{qms:.1f} ms**")
print(f"- gateway internal overhead: **{(e2e-eng)*1000:.1f} ms** ({(e2e-eng)/e2e*100:.1f}% of E2E)" if e2e else "- no data")
PY
    else
        echo "_no metrics scraped_"
    fi
} | tee "$SUMMARY"

echo
echo "results: $RESULTS_DIR"
