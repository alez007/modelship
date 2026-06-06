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
bench/run.sh                       # defaults: 100 prompts, conc 8, in/out 128/512
bench/run.sh --image modelship:dev --concurrency 32 --num-prompts 500
```

Tunable env vars (forwarded to the modelship phase):

- `MSHIP_GATEWAY_REPLICAS` (default 1) — gateway replica count.
- `MSHIP_GATEWAY_MAX_ONGOING` (default 1024) — gateway per-replica concurrency cap.
- `MSHIP_CACHE_DIR` — model cache to reuse across phases (default `./models-cache`).

The model and the per-model `max_ongoing_requests` cap come from
[`configs/bench.yaml`](configs/bench.yaml).

## Output

Each run writes a timestamped dir under `bench/results/` (gitignored) containing
`result.json` (per phase), `mem.tsv`, `prom.txt`, container logs, and a
`summary.md` with the comparison tables.

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
