# bench — modelship vs vanilla-loader A/B harness

Two-phase benchmark that runs `vllm bench serve` against a **modelship**
loader, then against the **vanilla server it wraps** (raw `vllm serve`, or a
bare `llama-server` subprocess) from the *same Docker image* with the same
engine config, and diffs throughput, latency, and memory.

This measures modelship's own wrapping overhead — Ray Serve, the gateway, the
loader's proxy layer — not one inference stack against another. `--loader`
picks which wrapped stack to measure (`vllm` or `llama_server`); `--device`
picks GPU vs CPU for that stack. The four combinations:

| `--loader` | `--device` | config | baseline |
| --- | --- | --- | --- |
| `vllm` | `gpu` (default) | `configs/vllm-gpu.yaml` | raw `vllm serve` on GPU |
| `vllm` | `cpu` | `configs/vllm-cpu.yaml` | raw `vllm serve` on the CPU backend |
| `llama_server` | `gpu` | `configs/llama-gpu.yaml` | vanilla `llama-server`, fully GPU-offloaded |
| `llama_server` | `cpu` | `configs/llama-cpu.yaml` | vanilla `llama-server`, CPU-only |

## Prerequisites

- `docker` and `curl` on the host.
- For `--device gpu`: the NVIDIA container runtime (`--gpus all` must work) and `nvidia-smi`.
- A built modelship image. Default tag is `modelship:dev` (GPU) / `modelship:dev-cpu`
  (CPU) — override with `--image`. GPU and CPU are separate image variants
  (`--build-arg MSHIP_VARIANT=gpu|cpu`, see the root `Dockerfile`); the `cpu`
  variant is required for `--device cpu` (it ships the `vllm==...+cpu` wheel).

## Run

```bash
bench/run.sh                                        # vllm on GPU (default): 100 prompts, conc 8, in/out 128/512, 20 warmups, 3 repeats
bench/run.sh --loader vllm --device cpu
bench/run.sh --loader llama_server --device gpu
bench/run.sh --loader llama_server --device cpu
bench/run.sh --loader vllm --image modelship:dev --concurrency 32 --num-prompts 500
```

`--num-warmups N` (default 20) sends warmup requests that are discarded before
timing so cold-start (CUDA graph capture / compilation / first-request JIT)
doesn't skew the result. `--repeats N` (default 3) runs the sweep N times per
stack; the summary reports the **median** so a single noisy run can't dominate.
`--config PATH` overrides the config file picked by `--loader`/`--device`.

Tunable env vars (forwarded to the modelship phase):

- `MSHIP_GATEWAY_REPLICAS` (default 1) — gateway replica count.
- `MSHIP_GATEWAY_MAX_ONGOING` (default 1024) — gateway per-replica concurrency cap.
- `MSHIP_CACHE_DIR` — model cache to reuse across phases (default `./models-cache`).

The model and the per-model `max_ongoing_requests` cap come from the config
file (`configs/vllm-gpu.yaml`, `configs/vllm-cpu.yaml`, `configs/llama-gpu.yaml`,
or `configs/llama-cpu.yaml`).

## Output

Each run writes a timestamped dir under `bench/results/` (gitignored) containing
`result_<n>.json` (one per repeat, per phase), `mem.tsv`, `prom.txt`,
`components.txt`, container logs, and a `summary.md` whose tables show the
**median across repeats**. The two phase subdirectories are always named
`modelship` and `baseline` regardless of `--loader`/`--device`.

`summary.md` also breaks the modelship container's memory down per Ray process
(from `components.txt`, scraped from the reporter agent on port 8079): a
**per-component memory** table ranks `ray::*` model-serving actors and the
control-plane processes (`gcs_server`, `raylet`, `agent`, `ProxyActor`,
`ServeController`) by private memory (USS), with shared *libraries* (torch/CUDA,
mapped into every worker — not plasma) reported separately so they aren't charged
to any one actor. This attributes the host-RAM overhead — model-serving actor vs
fixed Ray control plane. The snapshot is the **peak-private scrape sampled during
the sweep** (not the idle post-sweep state); modelship-only, since the baseline
stack has no Ray. Note this table **undercounts** the true total: the Ray
reporter sees only Ray worker PIDs, so a loader's own inference subprocess
(vLLM's `EngineCore`, or the `llama-server` child) is missing — the
reconciliation below quantifies the gap. Trust cgroup `anon` for the absolute
number.

Two cross-checks back this up:

- **cgroup `memory.stat` breakdown** — `mem.tsv` records, per second for *both*
  stacks, the kernel's own accounting: `anon` (real process RSS), `shmem`
  (tmpfs/plasma — Ray's object store, charged to the cgroup but to no process),
  and `file` (reclaimable page cache). The memory table reports the peak of each,
  so the modelship-vs-baseline RSS gap is attributed to real memory vs plasma vs cache.
- **reporter-vs-cgroup reconciliation** — the per-component section compares the
  Ray reporter's Σ private/shared (a second-hand Prometheus gauge that can be
  stale) against cgroup `anon`/`shmem` (ground truth). A `⚠️ diverges` flag means
  the reporter numbers are suspect and shouldn't be quoted. (vLLM and llama-server
  each expose their own `/metrics` too, but only engine stats — no per-process
  memory — which is why the cgroup numbers are the only cross-stack memory signal.)

## Results

Example run — 1×GPU, Qwen2.5-7B-AWQ, `--loader vllm --device gpu`, 100 prompts @ concurrency 8, median of 3:

| metric | modelship | raw vllm | overhead |
| --- | ---: | ---: | ---: |
| throughput (req/s) | 1.199 | 1.203 | −0.4% |
| output (tok/s) | 613.7 | 616.1 | −0.4% |
| TTFT mean (ms) | 62.5 | 54.9 | +13.9% |
| TTFT p95 (ms) | 89.5 | 65.0 | +37.8% |
| ITL mean (ms) | 12.96 | 12.39 | +4.6% |
| TPOT mean (ms) | 12.44 | 12.42 | +0.2% |
| peak VRAM (MiB) | 14533 | 14020 | +513 MiB |
| peak host RAM, anon (MiB) | 4957 | 3781 | **+1176 MiB** |

Notes:

- **Throughput and decode (TPOT) are at parity** — same vLLM wheel and GPU, so
  the engine's hot path is identical. modelship adds no per-token overhead.
- **TTFT/ITL** carry modelship's expected cost: the extra hop through the Ray
  Serve proxy/router adds a small *fixed* first-token latency (here ~7.6 ms) and a
  tiny per-chunk cost. Negligible for a 512-token response.
- **TTFT's tail is fatter than its mean**: p95 overhead (+37.8%) runs well above
  the mean/p50 gap (+13.9%/+1.6%) — occasional scheduling jitter through the Ray
  Serve proxy/router under concurrent load, not a fixed per-request cost. Still
  small in absolute terms (~25 ms) against a multi-second E2E latency.
- **Host RAM (+1176 MiB)** is the real resource cost — the Ray + Serve control
  plane (including the prewarmed idle-worker pool) plus the replica. This is `anon`
  (real process memory); **don't** use the container-RSS delta, which is dominated
  by reclaimable page cache and swings multiple GB between runs depending on which
  cgroup faulted the weights. VRAM overhead is modest (an extra CUDA context), not
  KV cache.
- Numbers are illustrative; they vary with model, hardware, load, loader, and device.

## How the two phases stay comparable

- Both phases use the same image (same vLLM wheel / same `llama-server` binary)
  and the same config file.
- The modelship phase runs with **`MSHIP_PREFLIGHT=false`** (passed via env). This is the linchpin: it disables hardware-aware automatic preflight tuning, ensuring that unset fields fall back to loader/pydantic defaults. Consequently, both phases run identical engine parameters out-of-the-box.
- The baseline phase parses the config through modelship's own pydantic schema
  and translates the engine config into either `vllm serve` flags
  ([`rawvllm_entrypoint.py`](rawvllm_entrypoint.py)) or a `llama-server` launch
  command ([`rawllama_entrypoint.py`](rawllama_entrypoint.py), mirroring
  `modelship/infer/llama_server/llama_server_infer.py`'s `_launch`).
- **Launch parity check**: After both phases run, the harness extracts each phase's effective launch command from its container logs, normalizes legitimately-different tokens (such as ports, hostnames, and api keys), and fails (exits non-zero) if there are any remaining differences. This guarantees that both phases run identical engine parameters.
- **Tokenizer extraction**: GGUF configs (which use GGUF model paths) cannot be used directly as Hugging Face repository IDs by the bench client. To handle this, the harness looks for a `# bench-tokenizer: <repo-id>` comment inside the yaml config file (inert to modelship) and parses it using `yaml_scalar` to use as the tokenizer for the bench client. You can also override it using the `--tokenizer` CLI flag.
- vLLM: keep `gpu_memory_utilization` equal to what modelship would pick for
  that `num_gpus` (0.9 GPU / 0.4 CPU) — for fractional `num_gpus` modelship
  overrides `gpu_memory_utilization` to `num_gpus`, and the raw phase reads the
  field verbatim, so a mismatch would make the comparison unfair.
- llama_server: set `n_gpu_layers` explicitly in `configs/llama-gpu.yaml`
  (rather than the loader's `-1` auto-fit default) — the raw phase has no
  preflight to pick a matching value on its own, so an explicit, identical
  value keeps both phases offloading the same number of layers.
- llama_server on a multi-GPU host: `rawllama_entrypoint.py` sets
  `CUDA_VISIBLE_DEVICES` to exactly `num_gpus` device(s) before exec'ing
  `llama-server`, mirroring the GPU reservation Ray gives the modelship
  actor. Without this the raw phase — a bare subprocess with no Ray actor —
  inherits every GPU the container's `--gpus` flag exposed, and llama.cpp
  auto-splits the model across all of them (no `--tensor-split`/`--main-gpu`
  is passed), handing the baseline more aggregate VRAM/bandwidth than the
  single-GPU modelship deploy and invalidating the comparison.
