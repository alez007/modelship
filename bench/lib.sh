# Shared helpers for the bench/run-*.sh A/B scripts. Sourced, not executed —
# assumes the caller has already set REPO_ROOT, BENCH_DIR, RESULTS_DIR,
# CACHE_DIR, and the sampler/cleanup PID vars it declares below.

# Extract the first scalar matching a key regex from a bench config yaml.
# Tolerates double-quoted, single-quoted, and unquoted values: strips
# everything up to the first colon, trims surrounding whitespace, then
# removes a matched pair of surrounding quotes (only when both ends use the
# same quote char).
yaml_scalar() {
    local pattern="$1" file="$2"
    grep -m1 -E "$pattern" "$file" \
        | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*\$//; s/^(['\"])(.*)\1\$/\2/" \
        || true
}

cleanup() {
    for pid in "${MEM_SAMPLER_PID:-}" "${COMPONENT_SAMPLER_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    for c in "$MODELSHIP_CONTAINER" "$BASELINE_CONTAINER"; do
        if [[ -n "$c" ]] && docker inspect "$c" >/dev/null 2>&1; then
            docker logs "$c" >"$RESULTS_DIR/${c}.log" 2>&1 || true
            docker rm -f "$c" >/dev/null 2>&1 || true
        fi
    done
}

wait_ready() {
    local name="$1"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        # /v1/models reachable AND lists the served model id
        local response
        if response=$(curl -fsS http://localhost:8000/v1/models 2>/dev/null); then
            if python3 -c "import sys, json; data = json.loads(sys.argv[1]); print('match' if any(m.get('id') == sys.argv[2] for m in data.get('data', [])) else '')" "$response" "$SERVED_NAME" | grep -q "match"; then
                return 0
            fi
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
            # sampling. Empty values fall back to 0 in the printf below. On a
            # CPU-only host nvidia-smi is simply absent, so vram stays 0 — the
            # sampler itself needs no device-aware branching.
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
# multi-minute phase. modelship-only — the baseline stack has no Ray reporter.
# Loader-agnostic: the Ray Serve control plane emits these regardless of which
# loader (vllm / llama_server) the deployment wraps. Router / request histograms
# are cumulative and still scraped once at the end.
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

# No-op (returns immediately) on a host with no nvidia-smi — there is no VRAM
# to wait on between phases on a CPU-only run.
vram_gate() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    local deadline=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < deadline )); do
        local used
        # tr -dc digits → "" when nvidia-smi errors or prints non-numeric
        # output; || true keeps pipefail+set -e from aborting the run. Guard
        # the arithmetic so an empty operand isn't a syntax error.
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9') || true
        if [[ -n "$used" ]] && (( used < 500 )); then return 0; fi
        sleep 1
    done
    echo "warn: VRAM not freed within 60s" >&2
}

# Read the model weights into the host page cache so both phases enter their
# timed sweeps equally warm. drop_host_caches can't drop without privileges (it
# warns and no-ops in most environments), and run.sh always runs modelship first
# / baseline second against the same GGUF — so the baseline would otherwise
# inherit a page cache the modelship phase had to cold-fault in. Called before
# each phase's sweeps: cheap (a no-op re-read once cached) and symmetric.
# Globs CACHE_DIR for *.gguf; on a cold host before the model is downloaded it
# simply finds nothing and returns, which is fine (that phase faults it in during
# its own model load, and the next phase is pre-warmed here).
#
# -L (dereference symlinks) is load-bearing: huggingface_hub stores the weights
# as a hash-named blob under blobs/ and exposes it via a snapshots/*.gguf
# *symlink*. Without -L, `-type f` skips the symlink and `-name '*.gguf'` misses
# the extensionless blob, so the glob matches nothing and the warm silently
# no-ops even with the model fully cached. With -L the .gguf symlink resolves to
# its blob and passes -type f. (2>/dev/null swallows the "No such file" find
# prints on any broken symlink.)
warm_model_cache() {
    echo "  pre-warming model cache..."
    local found=0
    while IFS= read -r -d '' f; do
        found=1
        cat "$f" > /dev/null 2>&1 || true
    done < <(find -L "$CACHE_DIR" -type f -name '*.gguf' -print0 2>/dev/null)
    if (( found )); then
        echo "  model cache warmed."
    else
        echo "  no .gguf found under cache dir yet — skipping warm."
    fi
}

drop_host_caches() {
    echo "  dropping host page caches..."
    if { sync && echo 3 > /proc/sys/vm/drop_caches; } >/dev/null 2>&1; then
        echo "  caches dropped successfully."
    elif sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1; then
        echo "  caches dropped successfully via sudo."
    else
        echo "  warn: failed to drop host caches (no write access and no passwordless sudo, or unsupported in this environment). Page-cache states may differ." >&2
    fi
}

# Run REPEATS timed sweeps against an already-ready stack, saving each to its
# own result_<n>.json. The summary takes the median across them.
run_stack() {
    local stack="$1"
    local out_dir="$RESULTS_DIR/$stack"
    mkdir -p "$out_dir"
    for i in $(seq 1 "$REPEATS"); do
        drop_host_caches
        echo "  sweep $i/$REPEATS ($stack)..."
        run_sweep "$stack" "result_${i}.json"
    done
}

run_sweep() {
    local stack="$1"
    local fname="$2"
    local out_dir="$RESULTS_DIR/$stack"

    local extra_client_args=()
    # E3. Disjoint cores for client vs server on `--device cpu`
    if [[ "$DEVICE" == "cpu" ]]; then
        local num_cores
        num_cores=$(nproc)
        if (( num_cores > 2 )); then
            local c_start=$(( num_cores - 2 ))
            local c_end=$(( num_cores - 1 ))
            extra_client_args+=(--cpuset-cpus "${c_start}-${c_end}")
            echo "  pinning client container to cpuset ${c_start}-${c_end} (of ${num_cores} cores)"
        else
            extra_client_args+=(--cpuset-cpus "0")
            echo "  pinning client container to cpuset 0"
        fi
    fi

    # --temperature 0 pins greedy decoding: vllm bench serve no longer forces it
    # (it warns and defers to the server default), so without this the two arms
    # sample independently and the comparison is nondeterministic run-to-run. With
    # --ignore-eos forcing generation past EOS, sampling also occasionally makes a
    # request emit a stray `<tool_call>` + malformed tool-call syntax that
    # llama-server's own grammar parser rejects mid-stream (an in-band 200-with-
    # error, propagated identically by both arms) — greedy makes that deterministic
    # and symmetric instead of landing randomly on one arm and reding the run.
    docker run --rm --network host --user "$(id -u):$(id -g)" \
        "${extra_client_args[@]}" \
        -v "$out_dir:/out:rw" "$IMAGE" \
        bash -lc "cd /modelship && uv run --active --no-sync vllm bench serve \
            --backend openai-chat \
            --base-url http://localhost:8000 \
            --endpoint /v1/chat/completions \
            --model $SERVED_NAME \
            --tokenizer $TOKENIZER \
            --dataset-name random \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $CONCURRENCY \
            --num-warmups $NUM_WARMUPS \
            --ignore-eos \
            --temperature 0 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,95,99 \
            --save-result \
            --save-detailed \
            --result-dir /out \
            --result-filename $fname"
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

assert_launch_parity() {
    echo "=== verifying launch-args parity ==="
    python3 - "$RESULTS_DIR" "$LOADER" <<'PY'
import sys, re, ast, shlex, os
from pathlib import Path

root = Path(sys.argv[1])
loader = sys.argv[2]

modelship_log = root / "bench-modelship.log"
baseline_log = root / "bench-baseline.log"

if not modelship_log.exists() or not baseline_log.exists():
    sys.exit(f"Logs missing. modelship log exists: {modelship_log.exists()}, baseline log exists: {baseline_log.exists()}")

m_content = modelship_log.read_text()
b_content = baseline_log.read_text()

def normalize_path(p):
    if not p:
        return ""
    if p.startswith("/") or "/" in p:
        return os.path.basename(p)
    return p

if loader == "llama_server":
    m_match = re.search(r"llama-server launch args for '.*':\s*(\[.*\])", m_content)
    if not m_match:
        sys.exit("Could not find 'llama-server launch args for' in modelship log")
    m_args = ast.literal_eval(m_match.group(1))

    b_match = re.search(r"rawllama exec:\s*(.*)", b_content)
    if not b_match:
        sys.exit("Could not find 'rawllama exec:' in baseline log")
    b_args = shlex.split(b_match.group(1))

    def normalize_llama_args(args):
        res = list(args[1:])
        normalized = []
        i = 0
        while i < len(res):
            arg = res[i]
            if arg in ["--host", "--port", "--api-key"]:
                i += 2
            elif arg in ["--alias"]:
                i += 2
            elif arg in ["-m", "--mmproj", "--chat-template-file", "--chat-template"]:
                if i + 1 < len(res):
                    normalized.append((arg, normalize_path(res[i+1])))
                    i += 2
                else:
                    normalized.append((arg, ""))
                    i += 1
            else:
                normalized.append((arg, ""))
                i += 1
        return sorted(normalized)

    m_norm = normalize_llama_args(m_args)
    b_norm = normalize_llama_args(b_args)

    with open(root / "launch-parity.txt", "w") as f:
        f.write(f"Modelship normalized: {m_norm}\n")
        f.write(f"Baseline normalized:  {b_norm}\n")

    if m_norm != b_norm:
        print("LAUNCH PARITY FAILED for llama_server!", file=sys.stderr)
        print(f"Modelship: {m_norm}", file=sys.stderr)
        print(f"Baseline:  {b_norm}", file=sys.stderr)
        sys.exit(1)
    else:
        print("LAUNCH PARITY PASSED for llama_server.")

elif loader == "vllm":
    m_match = re.search(r"initialising vllm engine with args:\s*(\{.*\})", m_content)
    if not m_match:
        sys.exit("Could not find 'initialising vllm engine with args:' in modelship log")
    m_dict = ast.literal_eval(m_match.group(1))

    b_match = re.search(r"rawvllm exec:\s*(.*)", b_content)
    if not b_match:
        sys.exit("Could not find 'rawvllm exec:' in baseline log")
    b_args = shlex.split(b_match.group(1))

    def parse_vllm_flags(args):
        parsed = {}
        flag_start = 0
        for idx, arg in enumerate(args):
            if arg.startswith('--'):
                flag_start = idx
                break
        
        i = flag_start
        while i < len(args):
            arg = args[i]
            if arg.startswith('--'):
                name = arg[2:].replace('-', '_')
                if name in ['enforce_eager', 'trust_remote_code', 'enable_auto_tool_choice']:
                    parsed[name] = True
                elif i + 1 < len(args):
                    val = args[i+1]
                    if val.isdigit():
                        parsed[name] = int(val)
                    else:
                        try:
                            parsed[name] = float(val)
                        except ValueError:
                            parsed[name] = val
                    i += 1
            i += 1
        return parsed

    b_dict = parse_vllm_flags(b_args)

    fields = [
        'gpu_memory_utilization',
        'tensor_parallel_size',
        'pipeline_parallel_size',
        'dtype',
        'quantization',
        'kv_cache_dtype',
        'enforce_eager',
        'trust_remote_code',
        'max_model_len',
    ]

    m_norm = {}
    b_norm = {}

    for fld in fields:
        mv = m_dict.get(fld)
        bv = b_dict.get(fld)
        if mv in [None, False]:
            mv = None
        if bv in [None, False]:
            bv = None
        if isinstance(mv, float) and isinstance(bv, float):
            if abs(mv - bv) < 1e-5:
                bv = mv
        m_norm[fld] = mv
        b_norm[fld] = bv

    with open(root / "launch-parity.txt", "w") as f:
        f.write(f"Modelship normalized: {m_norm}\n")
        f.write(f"Baseline normalized:  {b_norm}\n")

    if m_norm != b_norm:
        print("LAUNCH PARITY FAILED for vllm!", file=sys.stderr)
        print(f"Modelship: {m_norm}", file=sys.stderr)
        print(f"Baseline:  {b_norm}", file=sys.stderr)
        sys.exit(1)
    else:
        print("LAUNCH PARITY PASSED for vllm.")
PY
}

# Gate the run on result-population parity. The latency/throughput medians in
# summary.md are computed only over each arm's *successful* requests, with no
# check that both arms completed the same population — so if one arm silently
# drops its slowest requests as failures (e.g. the baseline's direct-to-
# cpp-httplib connections resetting under load), its tail metrics look better
# purely by survivorship. Fail loudly rather than publish a biased comparison.
assert_result_parity() {
    echo "=== verifying result-population parity (header + in-band SSE errors) ==="
    python3 - "$RESULTS_DIR" "$OUTPUT_LEN" <<'PY'
import json, sys
from pathlib import Path

# Two ways a request can fail, and the load client (vllm bench serve) only
# reliably catches one of them:
#
#   1. Header-level failure — the connection errors before/at the response
#      (e.g. baseline's cpp-httplib keep-alive resets → ServerDisconnectedError).
#      The client sets success=False and counts it in `failed`. Visible.
#
#   2. In-band failure — a streaming response whose HTTP 200 headers are already
#      flushed, then the body carries an OpenAI-style `data: {"error": ...}`
#      chunk followed by `[DONE]`. This is standard OpenAI streaming semantics
#      (you cannot downgrade a status once bytes are sent), which modelship
#      faithfully reproduces — and note the *error itself* often originates in
#      llama-server (e.g. its grammar parser rejecting malformed tool-call output
#      the model emits when --ignore-eos forces it past EOS), not in modelship's
#      wrapping. vllm bench serve's hand-rolled SSE parser only reads
#      `choices`/`usage` and silently skips the error chunk, so it counts the
#      request as `completed` with a *truncated* token stream. Invisible to
#      `failed`.
#
# Because the sweep runs with --ignore-eos, every healthy request emits exactly
# --random-output-len tokens. So a per-request output length below that (from
# --save-detailed's `output_lens`) is a hidden in-band failure — the only signal
# that survives an in-band error. We count it too, otherwise an arm that silently
# truncated N requests would still show completed==num_prompts and pass a
# survivorship-biased comparison the header check can't catch.
#
# Severity is RELATIVE between the two arms, because the bench's question is "does
# modelship cost anything *versus the raw server it wraps*?" — not "is either arm
# perfectly reliable". Both arms drive the *same* llama-server binary with the
# same greedy (--temperature 0) workload, so an in-band error that is really the
# engine's own (grammar rejection, etc.) shows up in both and is not a wrapping
# cost. Therefore:
#   * modelship drops/truncates MORE than baseline  → HARD FAIL (exit 1): a real
#     cost of the wrapper, and its medians compare unequal populations.
#   * baseline drops/truncates MORE than modelship  → FINDING (exit 0): a point in
#     modelship's favour (its uvicorn front door absorbs the cpp-httplib keep-alive
#     resets the raw server exposes). The baseline medians are then over its
#     surviving population — the completed/failed rows in summary.md flag that.
#   * equal (incl. both zero)  → PASS: any drops are shared workload/engine
#     behaviour, not attributable to the wrapper.
root = Path(sys.argv[1])
expected = int(sys.argv[2])

def scan(d):
    """Return (header_failed, hidden_inband, worst_partial_tok) for one result."""
    completed = d.get("completed", 0)
    failed = d.get("failed", 0)
    output_lens = d.get("output_lens")
    if output_lens:  # exact per-request path (needs --save-detailed)
        # Header failures already appended output_len 0, so subtract them to
        # isolate the hidden (200-with-error) failures miscounted as completed.
        short = sum(1 for ol in output_lens if ol < expected)
        hidden = max(0, short - failed)
        worst = min((ol for ol in output_lens if 0 < ol < expected), default=0)
        return failed, hidden, worst
    if completed and expected:  # aggregate fallback if arrays were stripped
        got = d.get("total_output_tokens", 0)
        full = completed * expected
        hidden = max(0, round((full - got) / expected)) if got < full else 0
        return failed, hidden, 0
    return failed, 0, 0

def summarize(stack):
    """Print per-sweep detail for one arm and return its (header, hidden) totals."""
    header_tot = hidden_tot = 0
    for p in sorted((root / stack).glob("result_*.json")):
        d = json.loads(p.read_text())
        completed = d.get("completed", 0)
        total = d.get("num_prompts", completed + d.get("failed", 0))
        failed, hidden, worst = scan(d)
        header_tot += failed
        hidden_tot += hidden
        msgs = []
        if failed:
            msgs.append(f"{failed} header FAILED / {completed} completed of {total}"
                        + ("  (cpp-httplib keep-alive resets)" if stack == "baseline" else ""))
        if hidden:
            partial = f", shortest partial {worst} tok" if worst else ""
            msgs.append(f"{hidden} HIDDEN in-band failure(s) — HTTP 200 but truncated "
                        f"below output_len={expected}{partial}, miscounted as completed")
        for m in msgs:
            print(f"  {stack}/{p.name}: {m}")
    return header_tot, hidden_tot

m_header, m_hidden = summarize("modelship")
b_header, b_hidden = summarize("baseline")
m_drops = m_header + m_hidden
b_drops = b_header + b_hidden
print(f"  totals: modelship {m_drops} dropped/truncated ({m_header} header + {m_hidden} in-band); "
      f"baseline {b_drops} ({b_header} header + {b_hidden} in-band)")

if m_drops > b_drops:
    sys.stdout.flush()  # keep the per-sweep detail (stdout) ahead of the verdict (stderr)
    print(
        f"\nRESULT PARITY FAILED: the modelship arm dropped or truncated MORE requests "
        f"than the raw baseline ({m_drops} vs {b_drops}). That excess is a cost of the "
        f"wrapper under test, and its latency/throughput medians compare unequal "
        f"populations (survivorship bias), so they are NOT trustworthy. Investigate "
        f"before trusting this run.",
        file=sys.stderr,
    )
    sys.exit(1)
if b_drops > m_drops:
    print()
    print("FINDING — baseline robustness gap (NOT a failure; run still passes):")
    print(
        f"The raw llama-server baseline dropped/truncated more requests than modelship "
        f"({b_drops} vs {m_drops}) under this load — largely the bench client hitting "
        f"cpp-httplib's keep-alive resets directly, which modelship's uvicorn front door "
        f"absorbs. This is a point in modelship's favour. Caveat: the baseline "
        f"latency/throughput medians in summary.md are computed over its surviving "
        f"requests only — read them as an upper bound on the baseline's advantage, not a "
        f"like-for-like population (see the completed/failed rows)."
        + (f" (modelship itself truncated {m_drops} request(s) — an in-band error it "
           f"shares with the baseline's engine, not a drop the baseline avoided.)"
           if m_drops else "")
    )
    sys.exit(0)
if m_drops:  # equal and non-zero
    print(
        f"\nRESULT PARITY PASSED: both arms dropped/truncated the same number of requests "
        f"({m_drops}) — shared workload/engine behaviour (e.g. llama-server rejecting "
        f"--ignore-eos-forced malformed tool calls), not a cost of the wrapper. The "
        f"populations match, so the medians are comparable."
    )
    sys.exit(0)
print("RESULT PARITY PASSED: both arms completed every request in full (no header or in-band errors).")
PY
}

# Renders the shared summary.md body: latency/throughput table (median across
# repeats), memory table (peak across the sweep), modelship per-component
# memory breakdown + cgroup cross-check, and the router-fulfillment Prometheus
# figure. $1/$2 are the stack directory names under $RESULTS_DIR (modelship
# phase first), used only as labels in the printed tables.
write_summary() {
    local modelship_stack="$1" baseline_stack="$2" baseline_label="$3"
    echo "| metric | modelship | $baseline_label | overhead |"
    echo "| --- | ---: | ---: | ---: |"
    python3 - "$RESULTS_DIR" "$modelship_stack" "$baseline_stack" <<'PY'
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
m = load(sys.argv[2]); r = load(sys.argv[3])
keys = [
    # Population first: latency/throughput below are over *successful* requests
    # only, so these two rows are the caveat for reading the rest of the table.
    # assert_result_parity fails the run when `failed` is non-zero on either arm,
    # so in a passing run both these are N and 0 respectively.
    ("completed",          "completed", 0),
    ("failed",             "failed", 0),
    ("request_throughput", "req/s", 3),
    ("output_throughput",  "output tok/s", 2),
    ("mean_ttft_ms",       "TTFT mean (ms)", 1),
    ("p50_ttft_ms",        "TTFT p50 (ms)", 1),
    ("p95_ttft_ms",        "TTFT p95 (ms)", 1),
    ("mean_tpot_ms",       "TPOT mean (ms)", 1),
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

# Token-count parity: prove both arms ran equivalent prompts. total_input_tokens
# is the client-side (shared --tokenizer) accounting; dividing by completed makes
# it robust to any success-count gap. modelship drains its llama-server subprocess
# logs at TRACE (suppressed in the bench container), so this reconciliation is the
# only independent check that the two arms tokenized the same work — the launch
# args being identical doesn't guarantee the prompt bodies were.
def per_prompt_in(runs):
    vals = [rr["total_input_tokens"] / rr["completed"] for rr in runs if rr.get("completed")]
    return statistics.median(vals) if vals else None
mi = per_prompt_in(m); ri = per_prompt_in(r)
if mi is not None and ri is not None:
    delta = mi - ri
    flag = "⚠️ prompts differ" if abs(delta) > 1.0 else "✓"
    print()
    print(f"_input tokens/prompt (client tokenizer): modelship **{mi:.1f}** vs "
          f"baseline **{ri:.1f}** (Δ {delta:+.1f}) {flag}_")
PY

    echo
    echo "## memory (peak across all sweeps)"
    python3 - "$RESULTS_DIR" "$modelship_stack" "$baseline_stack" "$baseline_label" <<'PY'
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
m = peak(sys.argv[2]); r = peak(sys.argv[3])
label = sys.argv[4]
print(f"| metric | modelship | {label} | overhead |")
print("| --- | ---: | ---: | ---: |")
def row(name, key, unit="MiB"):
    if m is None or r is None:
        return
    mv, rv = m[key], r[key]
    delta = mv - rv
    pct = f"{(delta / rv * 100):+.1f}%" if rv else "—"
    print(f"| {name} | {mv:.0f} {unit} | {rv:.0f} {unit} | {delta:+.0f} {unit} ({pct}) |")
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
    # we'd serve differently than the baseline) or the control plane (gcs_server /
    # raylet / agent / ProxyActor / ServeController — fixed Ray cost). USS is
    # private memory; shared is shared *libraries* (torch/CUDA, mapped into every
    # worker — NOT plasma) reported separately so it isn't charged to any one
    # actor. Snapshot is the peak-private scrape sampled *during* the sweep
    # (start_component_sampler), not the idle post-sweep state. The reconciliation
    # below shows this table undercounts the true total (misses non-Ray children).
    if [[ -s "$RESULTS_DIR/$modelship_stack/components.txt" ]]; then
        python3 - "$RESULTS_DIR" "$modelship_stack" <<'PY'
import sys, re
from pathlib import Path
root = Path(sys.argv[1])
pat = re.compile(r'^(ray_component_(?:uss_mb|rss_mb|mem_shared_bytes))\{([^}]*)\}\s+([0-9eE+.\-]+)')
key = {"ray_component_uss_mb": "uss", "ray_component_rss_mb": "rss", "ray_component_mem_shared_bytes": "shared"}
comp: dict[str, dict[str, float]] = {}
for line in (root / sys.argv[2] / "components.txt").read_text().splitlines():
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
    f = root / sys.argv[2] / "mem.tsv"
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
              "non-Ray child processes (notably the loader's own inference subprocess/engine) "
              "is missing. Trust the cgroup figure; treat the per-component split as "
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
    if [[ -s "$RESULTS_DIR/$modelship_stack/prom.txt" ]]; then
        python3 - "$RESULTS_DIR/$modelship_stack/prom.txt" <<'PY'
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
}
