# modelship Helm chart

Deploy [modelship](https://github.com/alez007/modelship) — an OpenAI-compatible,
multi-model inference server — on Kubernetes via [KubeRay](https://github.com/ray-project/kuberay).

The chart brings up a **RayCluster** (one CPU-only head + worker groups) and a
**RayJob** that submits `mship_deploy.py` to run **on** the cluster (KubeRay's
supported way to run a driver against a RayCluster) and deploy the models
declared in your `models.yaml`. Re-running (`helm upgrade`) re-applies the config
additively, or reconciles it when `deploy.reconcile=true`.

Each deploy persists this gateway's **effective config** (its desired model set)
and routing registry to a **state store** (see [Head-node HA](#head-node-ha-redis)),
so the gateway self-heals its routing after a head restart and `helm upgrade`
reconciles the live cluster back to the recorded set.

## Prerequisites

- A Kubernetes cluster (a local [kind](https://kind.sigs.k8s.io/) cluster works
  for the CPU image; GPU models need real GPU nodes with the NVIDIA device plugin).
- **The KubeRay operator + CRDs.** This is a cluster-scoped, install-once
  dependency. Either install it yourself:

  ```bash
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm install kuberay-operator kuberay/kuberay-operator
  ```

  …or, on a single-tenant cluster, let this chart bootstrap it:
  `--set kuberay-operator.enabled=true`.
- For GPU models: a node pool with `nvidia.com/gpu` resources.

## Install

```bash
# From the repo (path install):
helm install mship ./helm/modelship -f my-values.yaml

# From GHCR (OCI). The chart version is kept in lockstep with the app/image
# version, so --version <X.Y.Z> always pairs with the matching image:
helm install mship oci://ghcr.io/alez007/charts/modelship --version 0.4.0 -f my-values.yaml
```

Because images and model weights take time to pull, raise Helm's timeout:
`--timeout 20m --wait`. Note that `--wait` does **not** track the RayJob to
completion — watch `kubectl get rayjob` and the gateway `/readyz` for readiness.

## Configure your models

Set `models.config` to your `models.yaml` contents (see `config/examples/` in the
repo), or point at a ConfigMap you manage with `models.existingConfigMap`.

```yaml
models:
  config: |
    models:
      - name: qwen
        loader: vllm
        model: Qwen/Qwen2.5-7B-Instruct
        num_gpus: 1
```

Gated/private weights need a Hugging Face token; gateway auth needs API keys:

```yaml
secrets:
  huggingfaceToken: "hf_..."   # mounted as HF_TOKEN
  apiKeys: "sk-local-1,sk-local-2"
```

(Or reference an existing Secret with keys `HF_TOKEN` / `MSHIP_API_KEYS` via
`secrets.existingSecret`.)

## Topology

- **Head** — CPU-only (`num-gpus: 0`); runs GCS and the gateway. No models are
  scheduled here.
- **Serve HTTP proxies** — one runs on **every** Ray node (`proxy_location=EveryNode`),
  not just the head, and the gateway Service load-balances across all of them so
  ingress survives losing any single pod. Each proxy can route to any gateway
  replica wherever it's scheduled. Set `gateway.replicas > 1` (with ≥1 worker) for
  routing/ingress HA; replicas keep their routing tables in sync via the deploy
  coordinator.
- **Worker groups** — where models actually run. **Empty by default**, so a
  no-values install brings up only the head and schedules nothing; declare the
  groups that match your hardware under `workerGroups` (a commented gpu+cpu
  example ships in `values.yaml`). This is a **list — Helm replaces it wholesale**
  (no per-item merge), so always declare the full set you want; omitting the key
  keeps the empty default.
- **Cache** — a shared PVC for model weights at `/.cache`. Single-node clusters
  can use `ReadWriteOnce`; **multi-node requires `ReadWriteMany`** so every worker
  shares one copy.
- **/dev/shm** — an in-memory emptyDir (default 8Gi); vLLM/NCCL need it.

## Reaching the gateway

The gateway Service load-balances across the Serve proxy on every Ray node, gated
per-pod by proxy health (a pod only joins once its proxy is up). Check `/readyz`
for app-level readiness — it returns 503 until all models are loaded (use it for
an external LB/Ingress health check). Port-forward for local access, or set
`service.type=LoadBalancer`:

```bash
kubectl port-forward svc/<release>-modelship-gateway 8000:8000
curl http://localhost:8000/v1/models
```

## Head-node HA (Redis)

The head pod runs the GCS (Ray's control store). By default it's in-memory, so a
head restart (OOM, drain, eviction) loses cluster state: KubeRay recycles the
workers and every model has to be redeployed — minutes of full outage for a routine
reschedule. Set `redis.enabled=true` (pointing at a Redis you run) to close that gap.
One Redis backs two things at once:

1. **Ray GCS fault tolerance** (`gcsFaultToleranceOptions`) — a restarted head
   recovers GCS from Redis; workers and model actors **survive**, and Serve's
   controller redeploys anything that died. The restart becomes a sub-minute blip
   with no full redeploy.
2. **The modelship state store** (`MSHIP_STATE_STORE=redis://…`) — the deploy
   coordinator's routing registry and effective config live in Redis, so the gateway
   self-heals its routing on recovery instead of coming back empty.

```yaml
redis:
  enabled: true
  address: my-redis-master:6379
  password: "s3cret"          # or existingSecret + passwordKey
```

When `redis.enabled=false`, GCS stays in-memory and the state store falls back to
`file://` on the cache PVC — the effective config is still durable (so a manual
`helm upgrade` restores the model set after a full cluster loss), but live actors
don't survive a head restart. **What recovers automatically:**

| event | `redis.enabled=false` | `redis.enabled=true` |
|-------|----------------------|----------------------|
| head pod restart | full redeploy (workers recycled) | actors survive, routing self-heals — no redeploy |
| full cluster loss, Redis kept | `helm upgrade` | Serve + coordinator restore from Redis |
| full cluster loss, Redis also gone | `helm upgrade` | `helm upgrade` |

`externalStorageNamespace` is pinned to the release name so a recreated cluster
recovers; the password is injected via the Secret and expanded into the URI at
runtime, never landing in the pod manifest or argv. Bring your own Redis (a small
single instance with a PVC is plenty; the chart only wires an address).

## Common values

| Key | Default | Purpose |
|-----|---------|---------|
| `image.repository` / `image.tag` | `ghcr.io/alez007/modelship` / `<app version>` | Stamped to the release version |
| `image.variant` | `gpu` | `cpu` appends `-cpu` to the tag. GPU runs everywhere; set `cpu` on CPU-only clusters, or per worker group for a mixed cluster |
| `rayVersion` | `2.54.1` | Must match the Ray in the image |
| `models.config` / `models.existingConfigMap` | `models: []` | Your model set |
| `gateway.replicas` | `1` | API gateway replicas; raise (with ≥1 worker) for routing/ingress HA |
| `secrets.huggingfaceToken` / `secrets.apiKeys` | `""` | HF token / gateway API keys |
| `cache.size` / `cache.accessModes` | `100Gi` / `[ReadWriteOnce]` | Shared weight cache |
| `workerGroups` | `[]` | Worker pool layout (a list — set the full set; copy the example in `values.yaml`) |
| `deploy.reconcile` | `false` | Remove dropped models on upgrade |
| `deploy.replaceStrategy` | `blue_green` | How changed models are replaced |
| `redis.enabled` | `false` | Redis-backed GCS-FT + state store for head-node HA (see [Head-node HA](#head-node-ha-redis)) |
| `redis.address` | `""` | `host:port` of your Redis |
| `redis.password` / `redis.existingSecret` | `""` | Redis password inline, or reference an existing Secret (`passwordKey`) |
| `service.type` | `ClusterIP` | Set `LoadBalancer` to expose externally |
| `podMonitor.enabled` | `false` | Prometheus Operator scraping |
| `kuberay-operator.enabled` | `false` | Bootstrap the operator as a subchart |

See [values.yaml](values.yaml) for the full set with inline documentation.
