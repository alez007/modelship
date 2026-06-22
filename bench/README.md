# bench — modelship vs raw vLLM A/B harness

Two-phase benchmark that runs `vllm bench serve` against the **modelship** vLLM
loader, then against **raw `vllm serve`** from the *same Docker image* with the
same engine kwargs, and diffs throughput, latency, and memory.

## Prerequisites

- A GPU host with the NVIDIA container runtime (`--gpus all` must work).
- `docker`, `curl`, and `nvidia-smi` on the host.
- A built modelship image (default tag `modelship:dev`; override with `--image`).

## Run

```bash
bench/run.sh                       # defaults: 100 prompts, conc 8, in/out 128/512, 20 warmups, 3 repeats
bench/run.sh --image modelship:dev --concurrency 32 --num-prompts 500
```

`--num-warmups N` (default 20) sends warmup requests that are discarded before
timing so cold-start (CUDA graph capture / compilation) doesn't skew the result.
`--repeats N` (default 3) runs the sweep N times per stack; the summary reports
the **median** so a single noisy run can't dominate.

Tunable env vars (forwarded to the modelship phase):

- `MSHIP_GATEWAY_REPLICAS` (default 1) — gateway replica count.
- `MSHIP_GATEWAY_MAX_ONGOING` (default 1024) — gateway per-replica concurrency cap.
- `MSHIP_CACHE_DIR` — model cache to reuse across phases (default `./models-cache`).

The model and the per-model `max_ongoing_requests` cap come from
[`configs/bench.yaml`](configs/bench.yaml).

## Output

Each run writes a timestamped dir under `bench/results/` (gitignored) containing
`result_<n>.json` (one per repeat, per phase), `mem.tsv`, `prom.txt`,
`components.txt`, container logs, and a `summary.md` whose tables show the
**median across repeats**.

`summary.md` also breaks the modelship container's memory down per Ray process
(from `components.txt`, scraped from the reporter agent on port 8079): a
**per-component memory** table ranks `ray::*` model-serving actors and the
control-plane processes (`gcs_server`, `raylet`, `agent`, `ProxyActor`,
`ServeController`) by private memory (USS), with shared *libraries* (torch/CUDA,
mapped into every worker — not plasma) reported separately so they aren't charged
to any one actor. This attributes the host-RAM overhead — model-serving actor vs
fixed Ray control plane. The snapshot is the **peak-private scrape sampled during
the sweep** (not the idle post-sweep state); modelship-only, since raw vLLM has no
Ray. Note this table **undercounts** the true total: the Ray reporter sees only
Ray worker PIDs, so vLLM's `EngineCore` subprocess is missing — the reconciliation
below quantifies the gap. Trust cgroup `anon` for the absolute number.

Two cross-checks back this up:

- **cgroup `memory.stat` breakdown** — `mem.tsv` records, per second for *both*
  stacks, the kernel's own accounting: `anon` (real process RSS), `shmem`
  (tmpfs/plasma — Ray's object store, charged to the cgroup but to no process),
  and `file` (reclaimable page cache). The memory table reports the peak of each,
  so the modelship-vs-raw RSS gap is attributed to real memory vs plasma vs cache.
- **reporter-vs-cgroup reconciliation** — the per-component section compares the
  Ray reporter's Σ private/shared (a second-hand Prometheus gauge that can be
  stale) against cgroup `anon`/`shmem` (ground truth). A `⚠️ diverges` flag means
  the reporter numbers are suspect and shouldn't be quoted. (vLLM exposes a
  Prometheus `/metrics` too, but only engine stats — no per-process memory — which
  is why the cgroup numbers are the only cross-stack memory signal.)

## Results

Example run — 1×GPU, Qwen2.5-7B-AWQ, 100 prompts @ concurrency 8, median of 3:

| metric | modelship | raw vLLM | delta |
| --- | ---: | ---: | ---: |
| throughput (req/s) | 1.185 | 1.188 | −0.3% |
| output (tok/s) | 606.7 | 608.2 | −0.3% |
| TTFT mean (ms) | 71.3 | 63.4 | +8 ms |
| ITL mean (ms) | 12.99 | 12.53 | +0.5 ms |
| TPOT mean (ms) | 12.57 | 12.55 | ~0 |
| peak VRAM (MiB) | 14671 | 14070 | +601 |
| peak host RAM, anon (MiB) | 4598 | 3733 | **+865** |

Notes:

- **Throughput and decode (TPOT) are at parity** — same vLLM wheel and GPU, so
  the engine's hot path is identical. modelship adds no per-token overhead.
- **TTFT/ITL** carry modelship's expected cost: the extra hop through the Ray
  Serve proxy/router adds a small *fixed* first-token latency (here ~8 ms) and a
  tiny per-chunk cost. Negligible for a 512-token response.
- **Host RAM (~+0.9 GB)** is the real resource cost — the Ray + Serve control
  plane (including the prewarmed idle-worker pool) plus the replica. This is `anon`
  (real process memory); **don't** use the container-RSS delta, which is dominated
  by reclaimable page cache and swings multiple GB between runs depending on which
  cgroup faulted the weights. VRAM overhead is modest (an extra CUDA context), not
  KV cache.
- Numbers are illustrative; they vary with model, hardware, and load.

## How the two phases stay comparable

- Both phases use the same image (same vLLM wheel) and the same `bench.yaml`.
- The raw phase ([`rawvllm_entrypoint.py`](rawvllm_entrypoint.py)) parses
  `bench.yaml` through modelship's own pydantic schema and translates the
  vLLM engine kwargs into `vllm serve` flags.
- Keep `gpu_memory_utilization` equal to `num_gpus` in `bench.yaml`: for
  fractional `num_gpus` modelship overrides `gpu_memory_utilization` to
  `num_gpus`, and the raw phase reads the field verbatim — so a mismatch would
  make the comparison unfair.

> Caveat: modelship applies hardware-aware preflight defaults the raw phase does
> not, so this is a close A/B, not a byte-identical one.
