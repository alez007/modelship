# Production Readiness Plan

Future development priorities for making Modelship production-ready, organized by severity and area.

## Critical (Must Have Before Production)

### Security

- [x] **API authentication layer** — API key auth at the gateway level via `MSHIP_API_KEYS` env var; OpenAI-compatible `Authorization: Bearer <key>` header
- [ ] **Rate limiting** — per-user/IP/model throttling to prevent GPU resource monopolization
- [x] **Input size limits** — coarse payload size limit at the gateway (`MSHIP_MAX_REQUEST_BODY_BYTES`, default 50 MB); per-model `max_context_length` validation in every loader before inference
- [ ] **Lock down CORS** — replace wildcard `*` origins with environment-specific allowed origins
- [ ] **Plugin sandboxing** — plugins run with full server privileges; add signature verification or sandboxing

### Health & Readiness

- [x] **Detailed readiness probe** — `/readyz` returns 200 only when every expected model is registered with the gateway; 503 with loaded/pending lists while loading. `/health` stays as a cheap liveness endpoint. Per-model load times and total time-to-ready are exposed via `/readyz` for observability.
- [ ] **Model-specific health checks** — per-model liveness status (vLLM engine, Ray actor state)
- [ ] **GPU memory checks** — detect and report memory pressure before OOM

### Testing

- [x] **API endpoint tests** — HTTP-level tests for all `/v1/` endpoints (via `tests/test_integration.py`)
- [x] **Integration tests** — actual model loading and inference (using Qwen-0.5B/0.6B)
- [x] **Streaming tests** — SSE streaming correctness and error handling
- [x] **Plugin loading tests** — verify plugin lifecycle
- [ ] **Error recovery tests** — simulate failures and verify behavior

---

## High Priority (Should Have)

### Deployment & Infrastructure

- [x] **Kubernetes manifests** — KubeRay `RayCluster` + `RayJob`, gateway `Service`, models `ConfigMap`, cache `PVC`, secrets, optional `PodMonitor` (via the Helm chart in `helm/modelship`), with resource requests/limits, GPU scheduling, node affinity, and tolerations per worker group
- [x] **Helm chart** — parameterized deployment in `helm/modelship` (see its README)
- [ ] **Docker Compose** — for simpler non-K8s deployments
- [x] **Liveness/readiness probes in container spec** — KubeRay gates each Ray pod on a Serve proxy `/-/healthz` check (via the named `serve` port); `/readyz` returns 503 until all models load, suitable for an external LB/Ingress health check

### Alerting & Observability

- [x] **Prometheus alerting rules** — error rate thresholds, latency P99 breaches, model load failures, GPU memory pressure (see `docs/prometheus-alerts.yml`)
- [ ] **SLO/SLI definitions** — define target availability and latency for each endpoint type
- [x] **Structured logging (JSON)** — `MSHIP_LOG_FORMAT=json` for log aggregation (ELK/Loki/Splunk)
- [x] **Request-ID correlation** — trace a request from gateway through Ray actor boundaries via `contextvars`
- [x] **Log level configuration** — `MSHIP_LOG_LEVEL` controls app logs; `TRACE` enables library debug logs
- [x] **Syslog support** — `--log-target syslog://host:port` ships logs to a remote syslog server (UDP or TCP)
- [x] **OpenTelemetry log export** — `--otel-endpoint` ships logs (and enables Ray traces) via OTLP to any OTel collector

### Resilience

- [ ] **Ray actor restart policies** — auto-restart crashed model actors
- [ ] **Circuit breaker** — stop routing to a failing model after N consecutive errors
- [ ] **Backpressure / queue depth limits** — reject requests when queue is saturated instead of unbounded queuing
- [ ] **Graceful shutdown timeout** — add timeout wrapper around `serve.shutdown()` to prevent hanging
- [ ] **GPU OOM recovery** — detect and recover from GPU memory exhaustion

### Update Strategy

- [ ] **Rolling update support** — configure Ray Serve's built-in rolling updates for zero-downtime deploys
- [x] **Per-model autoscaling** — `autoscaling_config` (min/max replicas, target ongoing requests, up/downscale delays; scale-to-zero supported) scales replica count with load instead of a fixed `num_replicas`
- [x] **Gateway HA** — `MSHIP_GATEWAY_REPLICAS > 1` runs multiple gateway replicas; routing tables stay consistent via the deploy coordinator's watch loop, and a Serve proxy on every node lets the gateway Service survive single-pod loss
- [x] **Self-heal after cluster loss** — each deploy persists this gateway's effective config to the cache PVC; `mship_deploy --reconcile` (no `--config`) replays it, and the chart's `deploy.reassert` CronJob runs it periodically to restore the full model set if the cluster is recreated empty
- [x] **Model hot-reload** — allow `models.yaml` changes without full server restart (via `mship_deploy --reconcile`)
- [x] **Changelog** — track breaking changes between versions
- [x] **Migration guide** — document config format changes between versions

---

## Medium Priority (Nice to Have)

### CI/CD Hardening

- [ ] **Security scanning** — Trivy for Docker images, dependency vulnerability checks
- [ ] **SBOM generation** — Software Bill of Materials for supply chain visibility
- [ ] **Multi-arch builds** — ARM64 support alongside AMD64
- [ ] **Performance benchmarks in CI** — detect throughput/latency regressions

### Operations

- [ ] **Secrets management integration** — document Vault / K8s Secrets / sealed-secrets usage for `HF_TOKEN` and future API keys
- [ ] **Troubleshooting runbook** — common failure modes and resolution steps for on-call
- [ ] **Capacity planning guide** — estimate concurrent users per GPU setup per model mix
- [ ] **GPU memory budgeting guide** — model co-location recommendations to avoid fragmentation
- [x] **Multi-node Ray cluster setup docs** — head + worker topology, multi-node ingress, cache PVC access modes, and self-heal documented in the Helm chart README (`helm/modelship/README.md`)
- [ ] **Request audit trail** — persistent log of requests for compliance/debugging

### Documentation

- [ ] **OpenAPI/Swagger spec** — formal API reference for consumers
- [ ] **Performance tuning guide** — vLLM engine kwargs, batch sizes, KV cache sizing
- [ ] **Blue-green / canary deployment patterns** — documented strategies for safe rollouts
- [ ] **Model pre-warming** — mechanism to pre-download and cache models before deploy, reducing cold start from minutes to seconds

---

## Current Scorecard

| Area                         | Current | Target |
|------------------------------|---------|--------|
| Architecture & Design        | 9/10    | 9/10   |
| Monitoring (metrics)         | 9/10    | 9/10   |
| Monitoring (alerting + logs) | 9/10    | 9/10   |
| Security                     | 4/10    | 8/10   |
| Resilience                   | 7/10    | 8/10   |
| Testing                      | 8/10    | 9/10   |
| DevOps Experience            | 8/10    | 8/10   |
| Update/Deploy Strategy       | 7/10    | 7/10   |
