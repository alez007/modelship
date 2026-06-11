# bench — modelship vs raw vLLM A/B harness

Two-phase benchmark that runs `vllm bench serve` against the **modelship** vLLM
loader, then against **raw `vllm serve`** from the *same Docker image* with the
same engine kwargs, and diffs throughput, latency, and memory.

## Prerequisites

- A GPU host with the NVIDIA container runtime (`--gpus all` must work).
- `docker`, `curl`, and `nvidia-smi` on the host.
- A built modelship image (default tag `modelship:prod`; override with `--image`).

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
`result_<n>.json` (one per repeat, per phase), `mem.tsv`, `prom.txt`, container
logs, and a `summary.md` whose tables show the **median across repeats**.

## Results

Example run — 1×GPU, Qwen2.5-7B-AWQ, 100 prompts @ concurrency 8, median of 3:

| metric | modelship | raw vLLM | delta |
| --- | ---: | ---: | ---: |
| throughput (req/s) | 1.183 | 1.188 | −0.4% |
| output (tok/s) | 605.8 | 608.3 | −0.4% |
| TTFT mean (ms) | 81.2 | 64.1 | +17 ms |
| ITL mean (ms) | 12.91 | 12.52 | +0.4 ms |
| TPOT mean (ms) | 12.57 | 12.55 | ~0 |
| peak VRAM (MiB) | 14671 | 14070 | +601 |
| peak host RSS (MiB) | 8593 | 3925 | +4668 |

Notes:

- **Throughput and decode (TPOT) are at parity** — same vLLM wheel and GPU, so
  the engine's hot path is identical. modelship adds no per-token overhead.
- **TTFT/ITL** carry modelship's expected cost: the extra hop through the Ray
  Serve proxy/router adds a small *fixed* first-token latency (~tens of ms) and a
  tiny per-chunk cost. Negligible for a 512-token response (~3% of E2E here).
- **Host RAM (~+4.7 GB)** is the real resource cost — the Ray + Serve control
  plane. VRAM overhead is modest (an extra CUDA context), not KV cache.
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
