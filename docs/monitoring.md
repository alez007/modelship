# Monitoring & Logging

Modelship exposes Prometheus metrics through a single port via Ray's metrics agent. When enabled, all metrics — Ray cluster, Ray Serve, vLLM engine, and custom Modelship metrics — are available on one scrape endpoint.

## Logging

Modelship uses a centralized logging system with structured output and request correlation. All application logs go through the `modelship.*` logger hierarchy, separate from library logs (Ray, vLLM, etc.).

### Configuration

| Env Var | Default | Description |
|---|---|---|
| `MSHIP_LOG_LEVEL` | `INFO` | App log level. Set to `TRACE` for request/response payloads, `DEBUG` for detailed diagnostics. Each level sets library logs to the next level up (e.g. `DEBUG` app → `INFO` libs). |
| `MSHIP_LOG_FORMAT` | `text` | `text` for human-readable output, `json` for structured JSON lines (for log aggregation with ELK/Loki/Splunk). |
| `MSHIP_LOG_TARGET` | `console` | Log target. `console` writes to stderr; syslog URIs ship logs to a remote syslog server (see below). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | When set, logs are also exported to an OpenTelemetry collector via OTLP (see below). |

### Log Levels

Each level sets library logs (Ray, vLLM, transformers) to the next level up:

| Level | App logs (`modelship.*`) | Library logs |
|---|---|---|
| `TRACE` | Request/response payloads (audio bytes, transcription text, chat messages, etc.) | `DEBUG` |
| `DEBUG` | Detailed diagnostics, per-chunk details | `INFO` |
| `INFO` (default) | Startup, deployment, request summaries | `WARNING` |
| `WARNING` | Warnings only | `ERROR` |

### Request Correlation

Every API request is assigned a unique request ID that appears in all log lines for that request — both in the API gateway process and in the model deployment actor. This allows tracing a request end-to-end across Ray actor boundaries.

Text format example:
```
[2025-04-09 14:06:54] INFO     modelship.api [a1b2c3d4] | chat_completion model=llama messages=3 stream=True max_tokens=512
```

JSON format example:
```json
{"timestamp": "2025-04-09T14:06:54", "level": "INFO", "logger": "modelship.api", "message": "chat_completion model=llama messages=3 stream=True max_tokens=512", "request_id": "a1b2c3d4", "pid": 12345}
```

### Logger Names

| Logger | Scope |
|---|---|
| `modelship.startup` | Application initialization and shutdown |
| `modelship.api` | API gateway endpoints |
| `modelship.api.auth` | Authentication middleware |
| `modelship.infer` | Base inference layer |
| `modelship.infer.deployment` | Ray Serve model deployment actor |
| `modelship.infer.vllm` | vLLM inference backend |
| `modelship.infer.transformers` | Transformers inference backend |
| `modelship.infer.transformers.transcription` | Transformers speech-to-text/translation |
| `modelship.infer.transformers.chat` | Transformers chat/generation |
| `modelship.infer.transformers.embedding` | Transformers embeddings |
| `modelship.infer.transformers.speech` | Transformers TTS |
| `modelship.infer.diffusers` | Diffusers inference backend |
| `modelship.infer.diffusers.image` | Diffusers image generation |
| `modelship.infer.custom` | Custom/plugin inference backend |
| `modelship.plugin.<name>` | Individual plugins (kokoroonnx, bark, orpheus, whispercpp) |

### Syslog

Ship logs to a remote syslog server instead of stderr. Useful for centralized logging on bare-metal or Unraid setups without extra infrastructure.

```bash
# UDP (default)
python mship_deploy.py --log-target syslog://192.168.1.50:514

# TCP (reliable delivery)
python mship_deploy.py --log-target syslog+tcp://192.168.1.50:514

# Via environment variable
MSHIP_LOG_TARGET=syslog://192.168.1.50:514 python mship_deploy.py
```

Supported URI formats:

| URI | Protocol | Notes |
|---|---|---|
| `syslog://host:port` | UDP | Default, fire-and-forget |
| `syslog+tcp://host:port` | TCP | Reliable delivery |
| `syslog://host` | UDP | Port defaults to 514 |

The syslog target replaces the console handler — logs go to the syslog server only. The `--log-format` setting still applies (text or JSON).

### OpenTelemetry

Export logs to an OpenTelemetry collector via OTLP. Unlike syslog, OTel is additive — logs still go to the console (or syslog) handler, and are also shipped to the collector.

First, install the optional dependencies:

```bash
uv sync --extra otel
```

Then configure the endpoint:

```bash
# Via CLI
python mship_deploy.py --otel-endpoint http://collector:4317

# Via environment variable
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 python mship_deploy.py
```

When OTel is enabled:
- Logs are exported via `BatchLogRecordProcessor` with `OTLPLogExporter` (gRPC)
- The service name is set to `modelship`
- `RAY_TRACING_ENABLED=1` is set automatically so Ray workers also export traces
- HTTPS endpoints are detected from the URI scheme; all others use insecure connections

If the `opentelemetry-sdk` and `opentelemetry-exporter-otlp` packages are not installed, a warning is logged and OTel is skipped.

## Architecture

```
Prometheus  ──scrape──>  Ray Metrics Agent (:8079)
                              |
                              |-- ray_node_*          Ray cluster: GPU, CPU, memory
                              |-- ray_serve_*         Ray Serve: HTTP requests, latency, replicas
                              |-- ray_vllm_*          vLLM engine: KV cache, TTFT, tokens, queue
                              |-- ray_modelship_*         Custom: per-model latency, errors, load time
```

> **Note:** All metrics are prefixed with `ray_` by Ray's metrics agent. vLLM metric names are also sanitized (`:` → `_`), so e.g. the vLLM-native `vllm:kv_cache_usage_perc` becomes `ray_vllm_kv_cache_usage_perc`.

## Enabling Metrics

Metrics are enabled by default. Set `MSHIP_METRICS=false` to disable:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token \
  -e MSHIP_METRICS=true \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 -p 8079:8079 -p 8265:8265 \
  ghcr.io/alez007/modelship:latest
```

| Env Var | Default | Description |
|---|---|---|
| `MSHIP_METRICS` | `true` | Master toggle. Enables all metrics and the Ray metrics export port. |
| `RAY_METRICS_EXPORT_PORT` | `8079` | Port for the Ray metrics agent (only active when `MSHIP_METRICS=true`). |

Set `MSHIP_METRICS=false` to disable all metrics collection. When disabled, port 8079 is not exposed and there is zero overhead.

## Connecting to Prometheus

Add Modelship as a scrape target in your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: modelship
    scrape_interval: 15s
    static_configs:
      - targets: ["<modelship-host>:8079"]
```

For multi-node Ray clusters, use Ray's auto-generated service discovery file instead of static targets:

```yaml
scrape_configs:
  - job_name: modelship
    file_sd_configs:
      - files: ["/tmp/ray/prom_metrics_service_discovery.json"]
```

## Connecting to Grafana

A pre-built Grafana dashboard is included at [`docs/grafana-dashboard.json`](grafana-dashboard.json).

To import it:

1. Open Grafana and go to **Dashboards > Import**
2. Upload `grafana-dashboard.json` or paste its contents
3. Select your Prometheus datasource when prompted

The dashboard has 7 rows:

| Row | What it shows | Metric sources |
|---|---|---|
| **Overview** | Request rate, error rate, in-flight requests, models loaded, client disconnects | `ray_modelship_*` |
| **Latency** | Gateway P50/P95/P99, per-model latency, per-usecase latency (generate, TTS, image, STT, embed) | `ray_modelship_*` |
| **vLLM Engine** | KV cache usage, TTFT, inter-token latency, token throughput, queue depth, preemptions, prefix cache hit rate | `ray_vllm_*` |
| **GPU & System** | GPU utilization, GPU memory, CPU, system memory | `ray_node_*` |
| **Ray Serve** | Health check latency, request count, deployment processing latency, HTTP request latency | `ray_serve_*` |
| **Operational** | Model load time, load failures, resource cleanup errors, streaming chunks/s | `ray_modelship_*` |
| **Alerts** | Error rate %, KV cache usage, queue depth, TTFT P99, client disconnects, preemptions, GPU memory | `ray_modelship_*`, `ray_vllm_*`, `ray_node_*` |

## Alerting

A standalone Prometheus alerting rules file is included at [`docs/prometheus-alerts.yml`](prometheus-alerts.yml). The Grafana dashboard also has a dedicated **Alerts** row with threshold lines on the key panels.

### Importing Alert Rules

Add the rules file to your Prometheus config:

```yaml
rule_files:
  - /path/to/prometheus-alerts.yml
```

Then reload Prometheus (`kill -HUP <pid>` or `POST /-/reload` if `--web.enable-lifecycle` is set).

### Alert Reference

#### Critical (page-worthy)

| Alert | Condition | For | Description |
|---|---|---|---|
| `ModelshipHighErrorRate` | Error rate > 5% of traffic | 5m | Significant portion of requests are failing |
| `ModelshipNoModelsLoaded` | `models_loaded` == 0 | 2m | Server is running but cannot serve requests |
| `ModelshipModelLoadFailure` | Any increase in `model_load_failures_total` | 0m | A model failed to initialize |
| `ModelshipKVCacheExhausted` | KV cache usage > 95% | 5m | Requests will queue or be preempted |

#### Warning (investigate)

| Alert | Condition | For | Description |
|---|---|---|---|
| `ModelshipHighP99Latency` | Gateway P99 > 30s | 5m | End-to-end latency is very high |
| `ModelshipHighQueueDepth` | Waiting requests > 10 | 5m | vLLM engine is falling behind |
| `ModelshipPreemptions` | Preemption rate > 0 | 5m | GPU memory pressure causing request eviction |
| `ModelshipClientDisconnects` | Disconnect rate > 1/min | 5m | Clients timing out or dropping connections |
| `ModelshipGPUMemoryPressure` | Available GPU memory < 1 GB | 5m | GPU is nearly out of memory |
| `ModelshipHighTTFT` | TTFT P99 > 5s | 5m | Users waiting too long for first token |

### Tuning Thresholds

All thresholds are starting points. Adjust based on your deployment:

- **Error rate**: 5% is aggressive — if you run small models that occasionally OOM, raise to 10%.
- **P99 latency**: 30s works for chat completions with long outputs. For embeddings or TTS, consider lowering to 5-10s by adding per-endpoint rules.
- **Queue depth**: 10 assumes a single vLLM instance. Scale proportionally with replicas.
- **KV cache**: 95% is the danger zone. If you use prefix caching heavily, 90% may be more appropriate.
- **TTFT**: 5s is generous. For interactive chat, consider 2-3s.
- **GPU memory**: 1 GB threshold assumes you're not running anything else on the GPU. Raise if you have shared workloads.

## Health and readiness

Two endpoints are always available regardless of the metrics toggle.

**`/health`** — cheap liveness probe. Returns 200 as soon as the gateway is up (before model deployments finish):

```bash
curl http://localhost:8000/health
# {"status": "ok", "uptime_s": 12.3}
```

**`/readyz`** — readiness + timing. Returns 200 when every expected model has a registered deployment; 503 with the same JSON body while any model is still pending. Bodies carry full state so a single poll tells you what's loaded, what's outstanding, and how long each model took to come up:

```bash
curl http://localhost:8000/readyz
# 200 when ready:
# {
#   "status": "ok",
#   "ready": true,
#   "uptime_s": 420.5,
#   "time_to_ready_s": 215.3,
#   "models_loaded":   ["kokoro", "llm", "nomic-embed", "whisper"],
#   "models_expected": ["llm", "kokoro", "whisper", "nomic-embed"],
#   "models_pending":  [],
#   "model_load_times_s": {"llm": 50.2, "kokoro": 8.1, "whisper": 120.5, "nomic-embed": 36.5}
# }
```

Per-model timings are gateway-measured: the gap between one model registering and the next (models deploy sequentially in `mship_deploy.py`), so the first model's entry includes any framework-level setup time preceding it.

Use `/health` for Kubernetes liveness probes and `/readyz` for readiness probes — `/readyz` returning 503 prevents a service from flipping traffic onto the pod before models are loaded.

## Modelship Metrics Reference

All custom metrics are defined via `ray.serve.metrics` and exported through Ray's metrics agent with a `ray_` prefix.

### Gateway

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_request_total` | Counter | `model`, `endpoint`, `status` | Total requests by model and API method |
| `ray_modelship_request_duration_seconds` | Histogram | `model`, `endpoint` | End-to-end request latency |
| `ray_modelship_request_errors_total` | Counter | `model`, `endpoint`, `error_type` | Errors: `inference_error`, `stream_error`, `unhandled` |
| `ray_modelship_request_in_progress` | Gauge | `model`, `endpoint` | Currently processing requests |
| `ray_modelship_client_disconnects_total` | Counter | `model`, `endpoint` | Client disconnected before response completed |
| `ray_modelship_stream_chunks_total` | Counter | `model` | Streaming chunks emitted |

### Model Deployment

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_model_load_duration_seconds` | Histogram | `model`, `loader` | Time to initialize a model |
| `ray_modelship_model_load_failures_total` | Counter | `model`, `loader` | Failed model initializations |
| `ray_modelship_models_loaded` | Gauge | | Number of loaded and ready models |

### Inference Timing

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_generation_duration_seconds` | Histogram | `model` | Chat/text generation latency |
| `ray_modelship_tts_generation_duration_seconds` | Histogram | `model` | Text-to-speech latency |
| `ray_modelship_image_generation_duration_seconds` | Histogram | `model` | Image generation latency |
| `ray_modelship_transcription_duration_seconds` | Histogram | `model` | Speech-to-text latency |
| `ray_modelship_embedding_duration_seconds` | Histogram | `model` | Embedding latency |

### Resource Cleanup

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_resource_cleanup_errors_total` | Counter | `model`, `component` | Errors during engine/model cleanup |

## Built-in Metrics from vLLM and Ray

These are automatically available when `MSHIP_METRICS=true` — no additional configuration needed.

### vLLM (`ray_vllm_*`)

vLLM metrics are routed through Ray's metrics agent via `RayPrometheusStatLogger`. The native `vllm:` prefix is sanitized to `ray_vllm_`.

- `ray_vllm_num_requests_running` / `ray_vllm_num_requests_waiting` — queue depth
- `ray_vllm_kv_cache_usage_perc` — KV cache utilization (0-1)
- `ray_vllm_time_to_first_token_seconds` — TTFT histogram
- `ray_vllm_inter_token_latency_seconds` — ITL histogram
- `ray_vllm_e2e_request_latency_seconds` — end-to-end latency histogram
- `ray_vllm_request_queue_time_seconds` — time spent waiting in queue
- `ray_vllm_prompt_tokens_total` / `ray_vllm_generation_tokens_total` — token throughput counters
- `ray_vllm_num_preemptions_total` — memory pressure signal
- `ray_vllm_prefix_cache_hits_total` / `ray_vllm_prefix_cache_queries_total` — cache efficiency

Full reference: [vLLM Metrics Documentation](https://docs.vllm.ai/en/stable/design/metrics/)

### Ray Serve (`ray_serve_*`)

- `ray_serve_num_http_requests_total` — request count by route, method, status
- `ray_serve_http_request_latency_ms` — request latency histogram
- `ray_serve_handle_request_counter_total` — request count by deployment
- `ray_serve_deployment_processing_latency_ms` — per-replica processing time
- `ray_serve_health_check_latency_ms` — health check latency histogram

Full reference: [Ray Serve Monitoring](https://docs.ray.io/en/latest/serve/monitoring.html)

### Ray Cluster (`ray_*`)

- `ray_node_gpus_utilization` — GPU utilization by device
- `ray_node_gram_used` / `ray_node_gram_available` — GPU memory
- `ray_node_cpu_utilization` — CPU usage
- `ray_node_mem_used` / `ray_node_mem_total` — system memory

Full reference: [Ray Metrics](https://docs.ray.io/en/latest/cluster/metrics.html)
