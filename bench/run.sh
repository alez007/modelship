#!/usr/bin/env bash
# A/B benchmark: modelship vLLM loader vs raw vLLM, same image, same config.
# Usage: bench/run.sh [--image TAG] [--num-prompts N] [--concurrency N] [--input-len N] [--output-len N]
#                     [--num-warmups N] [--repeats N]
set -euo pipefail

IMAGE="${IMAGE:-modelship:dev}"
NUM_PROMPTS=100
CONCURRENCY=8
INPUT_LEN=128
OUTPUT_LEN=512
# Warmup requests sent (and discarded) before timing begins. Warms CUDA graph
# capture / lazy compilation so the first few real requests don't eat a
# multi-second cold-start tail that skews mean/p99 TTFT and throughput.
NUM_WARMUPS=20
# Timed sweeps per stack. We report the median across repeats so a single cold
# or noisy run can't dominate; tail metrics on one ~100-prompt run are unstable.
REPEATS=3
READY_TIMEOUT=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --num-warmups) NUM_WARMUPS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
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
    for pid in "${MEM_SAMPLER_PID:-}" "${COMPONENT_SAMPLER_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
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
            local ts vram cmem cgstat amib fmib smib
            ts=$(date +%s)
            # || true: under pipefail+set -e a failing nvidia-smi/docker stats
            # would otherwise abort this backgrounded subshell and silently stop
            # sampling. Empty values fall back to 0 in the printf below.
            vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') || true
            # Mem usage like "1.234GiB / 64GiB" — take first field, normalize to
            # MiB. Handle binary (GiB/MiB/KiB) and decimal (GB/MB/KB) units,
            # any case, with or without a space before the unit; unknown units
            # fall back to assuming the value is already in MiB.
            cmem=$(docker stats --no-stream --format '{{.MemUsage}}' "$container" 2>/dev/null \
                | awk -F'/' '{print $1}' \
                | awk '{
                    s=$0; gsub(/[[:space:]]/,"",s);   # e.g. "1.234GiB"
                    num=s; unit=s;
                    sub(/[A-Za-z]+$/,"",num);         # numeric part
                    sub(/^[0-9.]+/,"",unit);          # unit part
                    U=toupper(unit);
                    base=(U ~ /I/)?1024:1000;         # *iB binary, *B decimal
                    p=substr(U,1,1);
                    if      (p=="T") mib=num*base*base*base*base/1048576;
                    else if (p=="G") mib=num*base*base*base/1048576;
                    else if (p=="M") mib=num*base*base/1048576;
                    else if (p=="K") mib=num*base/1048576;
                    else if (U=="B") mib=num/1048576;
                    else             mib=num;         # unknown/unitless: assume MiB
                    printf "%.1f", mib
                  }') || true
            # cgroup memory.stat breakdown (bytes→MiB), sampled for *both* stacks
            # so the peak-RSS gap can be attributed: anon = real process RSS,
            # shmem = tmpfs/plasma (Ray's object store — charged to the cgroup but
            # to no single process, so the per-component table never sees it),
            # file = reclaimable page cache. cgroup v2 path; if absent (v1/missing)
            # the awk END still emits zeros.
            cgstat=$(docker exec "$container" cat /sys/fs/cgroup/memory.stat 2>/dev/null) || true
            read -r amib fmib smib < <(printf '%s\n' "$cgstat" | awk '
                $1=="anon"{a=$2} $1=="file"{f=$2} $1=="shmem"{s=$2}
                END {printf "%.1f %.1f %.1f", a/1048576, f/1048576, s/1048576}') || true
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${ts:-0}" "${vram:-0}" "${cmem:-0}" "${amib:-0}" "${fmib:-0}" "${smib:-0}" >> "$out"
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

# Sample the Ray reporter's per-component memory (port 8079) *during* the sweep
# and keep the scrape with the highest total private memory, so the breakdown
# reflects peak load instead of the idle post-sweep state. The reporter refreshes
# its gauges on its own interval; polling at 2s catches every refresh over a
# multi-minute phase. modelship-only — raw vLLM has no Ray reporter. Router /
# request histograms are cumulative and still scraped once at the end.
start_component_sampler() {
    local out="$1"
    : > "$out"
    (
        local best=-1 comp score
        while :; do
            # || true: a failed scrape under pipefail+set -e must not kill the loop.
            comp=$(curl -fsS http://localhost:8079/metrics 2>/dev/null \
                | awk '/^ray_component_(uss_mb|rss_mb|mem_shared_bytes)[{ ]/') || true
            if [[ -n "$comp" ]]; then
                # Score = total private (USS) across components. $NF is the metric
                # value — robust to spaces inside Component label values (e.g.
                # "ray::ServeReplica:modelship api:modelship api").
                score=$(printf '%s\n' "$comp" \
                    | awk '/^ray_component_uss_mb[{ ]/ {s+=$NF} END {printf "%.0f", s+0}')
                if [[ -n "$score" ]] && (( score > best )); then
                    best=$score
                    printf '%s\n' "$comp" > "$out"
                fi
            fi
            sleep 2
        done
    ) &
    COMPONENT_SAMPLER_PID=$!
}

stop_component_sampler() {
    if [[ -n "${COMPONENT_SAMPLER_PID:-}" ]] && kill -0 "$COMPONENT_SAMPLER_PID" 2>/dev/null; then
        kill "$COMPONENT_SAMPLER_PID" 2>/dev/null || true
        wait "$COMPONENT_SAMPLER_PID" 2>/dev/null || true
    fi
    COMPONENT_SAMPLER_PID=""
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
    local fname="$2"
    local out_dir="$RESULTS_DIR/$stack"
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
            --num-warmups $NUM_WARMUPS \
            --ignore-eos \
            --save-result \
            --result-dir /out \
            --result-filename $fname"
}

# Run REPEATS timed sweeps against an already-ready stack, saving each to its
# own result_<n>.json. The summary takes the median across them.
run_stack() {
    local stack="$1"
    local out_dir="$RESULTS_DIR/$stack"
    mkdir -p "$out_dir"
    chmod 777 "$out_dir"
    for i in $(seq 1 "$REPEATS"); do
        echo "  sweep $i/$REPEATS ($stack)..."
        run_sweep "$stack" "result_${i}.json"
    done
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
    # Router / request histograms only — these are cumulative counters, so a
    # single end-of-sweep scrape is correct. Per-component *memory* is a gauge
    # that varies with load and is captured under load by start_component_sampler,
    # not here. || true: an empty scrape must not abort the run under pipefail.
    curl -fsS http://localhost:8079/metrics 2>/dev/null \
        | awk '/^ray_modelship_(request|generation)_duration_seconds_(sum|count)/ \
              || /^ray_serve_request_router_fulfillment_time_ms_(sum|count)/' \
        > "$out" || true
}

echo "=== bench $TS — image=$IMAGE prompts=$NUM_PROMPTS conc=$CONCURRENCY in=$INPUT_LEN out=$OUTPUT_LEN warmups=$NUM_WARMUPS repeats=$REPEATS ==="

# Phase A — modelship
echo "[A] starting modelship..."
start_modelship
wait_ready bench-modelship
echo "[A] running $REPEATS sweep(s)..."
mkdir -p "$RESULTS_DIR/modelship"
start_mem_sampler modelship bench-modelship
start_component_sampler "$RESULTS_DIR/modelship/components.txt"
run_stack modelship
stop_mem_sampler
stop_component_sampler
scrape_prom "$RESULTS_DIR/modelship/prom.txt"
docker rm -f bench-modelship >/dev/null
vram_gate

# Phase B — rawvllm
echo "[B] starting rawvllm..."
start_rawvllm
wait_ready bench-rawvllm
echo "[B] running $REPEATS sweep(s)..."
mkdir -p "$RESULTS_DIR/rawvllm"
start_mem_sampler rawvllm bench-rawvllm
run_stack rawvllm
stop_mem_sampler
docker rm -f bench-rawvllm >/dev/null

# Summary
SUMMARY="$RESULTS_DIR/summary.md"
{
    echo "# bench $TS"
    echo
    echo "image: \`$IMAGE\`  prompts: $NUM_PROMPTS  concurrency: $CONCURRENCY  input/output: $INPUT_LEN/$OUTPUT_LEN  warmups: $NUM_WARMUPS  repeats: $REPEATS"
    echo
    echo "Values are the median across \`repeats\` sweeps."
    echo
    echo "| metric | modelship | rawvllm | overhead |"
    echo "| --- | ---: | ---: | ---: |"
    python3 - "$RESULTS_DIR" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
def load(stack):
    # One result_<n>.json per repeat; return them all so we can take medians.
    runs = [json.loads(p.read_text()) for p in sorted((root / stack).glob("result_*.json"))]
    if not runs:
        sys.exit(f"no result_*.json found for {stack}")
    return runs
def med(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(vals) if vals else None
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
    mv = med(m, key); rv = med(r, key)
    if mv is None or rv is None:
        continue
    if rv == 0:
        ratio = "—"
    else:
        ratio = f"{(mv - rv) / rv * 100:+.1f}%"
    print(f"| {label} | {mv:.{prec}f} | {rv:.{prec}f} | {ratio} |")
PY
    echo
    echo "## memory (peak across all sweeps)"
    python3 - "$RESULTS_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
# mem.tsv columns: ts, vram, container_rss, anon, file(cache), shmem — all MiB
# except ts. Older runs only have the first 3; missing columns peak at 0.
COLS = ["vram", "rss", "anon", "file", "shmem"]
def peak(stack):
    f = root / stack / "mem.tsv"
    if not f.exists():
        return None
    peaks = {c: 0.0 for c in COLS}
    for line in f.read_text().splitlines():
        parts = line.split("\t")
        for i, c in enumerate(COLS, start=1):
            if i < len(parts):
                try:
                    peaks[c] = max(peaks[c], float(parts[i]))
                except ValueError:
                    pass
    return peaks
m = peak("modelship"); r = peak("rawvllm")
print("| metric | modelship | rawvllm | overhead |")
print("| --- | ---: | ---: | ---: |")
def row(label, key, unit="MiB"):
    if m is None or r is None:
        return
    mv, rv = m[key], r[key]
    delta = mv - rv
    pct = f"{(delta / rv * 100):+.1f}%" if rv else "—"
    print(f"| {label} | {mv:.0f} {unit} | {rv:.0f} {unit} | {delta:+.0f} {unit} ({pct}) |")
row("peak VRAM (GPU0)", "vram")
row("peak container RSS", "rss")
row("  ├─ anon (process RSS)", "anon")
row("  ├─ shmem (tmpfs/plasma)", "shmem")
row("  └─ file (page cache)", "file")
print()
print("_**anon** is the real RAM overhead. **file** (page cache) is reclaimable "
      "and non-deterministic — it depends on which cgroup first faulted the weights "
      "and can swing GB between runs, so the container-RSS delta over- or "
      "under-states the true cost. Each row is an independent peak (different "
      "instants), so the sub-rows need not sum to peak RSS._")
PY
    echo
    echo "## modelship per-component memory (Ray reporter, peak under load)"
    echo
    # Breaks the modelship container's RSS down by Ray process so we can see
    # whether the overhead lives in the model-serving actor (ray::*Deployment* —
    # we'd serve differently than raw vLLM) or the control plane (gcs_server /
    # raylet / agent / ProxyActor / ServeController — fixed Ray cost). USS is
    # private memory; shared is shared *libraries* (torch/CUDA, mapped into every
    # worker — NOT plasma) reported separately so it isn't charged to any one
    # actor. Snapshot is the peak-private scrape sampled *during* the sweep
    # (start_component_sampler), not the idle post-sweep state. The reconciliation
    # below shows this table undercounts the true total (misses non-Ray children).
    if [[ -s "$RESULTS_DIR/modelship/components.txt" ]]; then
        python3 - "$RESULTS_DIR" <<'PY'
import sys, re
from pathlib import Path
root = Path(sys.argv[1])
pat = re.compile(r'^(ray_component_(?:uss_mb|rss_mb|mem_shared_bytes))\{([^}]*)\}\s+([0-9eE+.\-]+)')
key = {"ray_component_uss_mb": "uss", "ray_component_rss_mb": "rss", "ray_component_mem_shared_bytes": "shared"}
comp: dict[str, dict[str, float]] = {}
for line in (root / "modelship" / "components.txt").read_text().splitlines():
    m = pat.match(line)
    if not m:
        continue
    metric, labels, val = m.group(1), m.group(2), float(m.group(3))
    name = dict(re.findall(r'(\w+)="([^"]*)"', labels)).get("Component", "?")
    d = comp.setdefault(name, {})
    # Ray emits rss/uss in MB (bytes/1e6); shared is raw bytes — normalize to MB.
    v = val / 1e6 if metric == "ray_component_mem_shared_bytes" else val
    d[key[metric]] = d.get(key[metric], 0.0) + v
# Private = USS when the agent could read it, else RSS - shared as a floor.
def private(d):
    return d["uss"] if "uss" in d else max(d.get("rss", 0.0) - d.get("shared", 0.0), 0.0)
rows = sorted(comp.items(), key=lambda kv: private(kv[1]), reverse=True)
print("| component | private (MB) | rss (MB) | shared (MB) |")
print("| --- | ---: | ---: | ---: |")
tot_priv = tot_rss = tot_shared = 0.0
for name, d in rows:
    p, r, s = private(d), d.get("rss", 0.0), d.get("shared", 0.0)
    tot_priv += p; tot_rss += r; tot_shared += s
    print(f"| `{name}` | {p:.0f} | {r:.0f} | {s:.0f} |")
print(f"| **total** | **{tot_priv:.0f}** | **{tot_rss:.0f}** | **{tot_shared:.0f}** |")
if not any("uss" in d for _, d in rows):
    print()
    print("_USS unavailable (reporter couldn't read smaps); private column is "
          "rss − shared, an upper bound._")

# Cross-check the Ray reporter (Prometheus) against the kernel's own accounting
# (cgroup memory.stat, peak under load). The reporter samples /proc smaps on its
# own interval and can be stale or miss workers; cgroup is ground truth. anon ≈
# Σ private, shmem ≈ Σ shared. Peaks are sampled independently so expect rough,
# not exact, agreement — a large gap means the per-component table is suspect.
def cgroup_peak(col):  # mem.tsv: ts,vram,rss,anon,file,shmem (MiB)
    f = root / "modelship" / "mem.tsv"
    if not f.exists():
        return None
    peak = 0.0
    for line in f.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > col:
            try:
                peak = max(peak, float(parts[col]))
            except ValueError:
                pass
    return peak
anon = cgroup_peak(3)  # MiB ≈ MB for this sanity check
if anon:
    print()
    gap = (tot_priv - anon) / anon * 100
    flag = "  ⚠️ diverges" if abs(gap) > 25 else ""
    print("_Reporter cross-check vs cgroup `memory.stat` (kernel ground truth, peak):_")
    print(f"- private: reporter Σ USS **{tot_priv:.0f} MB** vs cgroup anon "
          f"**{anon:.0f} MB** ({gap:+.0f}%){flag}")
    if gap < -25:
        print("  - reporter undercounts — it sees only Ray worker PIDs, so memory in "
              "non-Ray child processes (notably vLLM's `EngineCore` subprocess) is "
              "missing. Trust the cgroup figure; treat the per-component split as "
              "relative attribution, not an absolute total.")
    # NOTE: Ray's mem_shared_bytes is shared *libraries* (torch/CUDA, PSS-shared),
    # not plasma — so it has no clean cgroup counterpart and is deliberately not
    # reconciled. Actual tmpfs/plasma is cgroup `shmem` (see the memory table); it
    # is tiny here, confirming the streaming path barely touches the object store.
PY
    else
        echo "_no component metrics scraped (reporter agent down or 8079 unreachable)_"
    fi
    echo
    echo "## modelship internal (Prometheus)"
    echo
    # NOTE: modelship_request_duration_seconds and modelship_generation_duration_seconds
    # are observed when the streaming generator is *created*, not after it drains
    # (model_deployment.py:230, api.py:347), so for streaming responses they capture
    # setup/TTFT only — not end-to-end or full-generation time. We therefore do NOT
    # derive "gateway overhead" from them (it's meaningless and was wildly wrong).
    # Only the router fulfillment time below reflects real request handling.
    if [[ -s "$RESULTS_DIR/modelship/prom.txt" ]]; then
        python3 - "$RESULTS_DIR/modelship/prom.txt" <<'PY'
import sys, re
sums = {}; counts = {}
pat = re.compile(
    r'(ray_serve_request_router_fulfillment_time_ms)'
    r'_(sum|count)\S*\s+([0-9eE+\-.]+)'
)
for line in open(sys.argv[1]):
    m = pat.match(line)
    if not m: continue
    name, kind, val = m.group(1), m.group(2), float(m.group(3))
    (sums if kind == "sum" else counts).setdefault(name, 0.0)
    if kind == "sum": sums[name] += val
    else: counts[name] += val
n = "ray_serve_request_router_fulfillment_time_ms"
cnt = counts.get(n, 0.0)
if cnt:
    print(f"- mean router fulfillment (routing + queue wait): **{sums.get(n, 0.0) / cnt:.1f} ms** "
          f"over {cnt:.0f} routed requests")
else:
    print("- no router metrics scraped")
print()
print("_E2E / engine durations omitted: their histograms are recorded before "
      "streaming completes and are not meaningful for streaming responses._")
PY
    else
        echo "_no metrics scraped_"
    fi
} | tee "$SUMMARY"

echo
echo "results: $RESULTS_DIR"
