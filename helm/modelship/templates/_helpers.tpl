{{/*
Chart name, optionally overridden by nameOverride.
*/}}
{{- define "modelship.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name. Honors fullnameOverride; otherwise release-name based.
*/}}
{{- define "modelship.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "modelship.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "modelship.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: modelship
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "modelship.selectorLabels" -}}
app.kubernetes.io/name: {{ include "modelship.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Effective image tag, resolving the gpu/cpu variant onto the base tag. `variant: cpu` appends "-cpu";
an explicit `tag`/`variant` (e.g. a per-worker-group override) wins over the
cluster-wide image values. Call with a dict: (dict "root" $) or
(dict "root" $ "tag" $img.tag "variant" $img.variant).
*/}}
{{- define "modelship.imageTag" -}}
{{- $tag := .tag | default .root.Values.image.tag -}}
{{- $variant := .variant | default .root.Values.image.variant | default "cuda" -}}
{{- if eq $variant "cpu" -}}{{ printf "%s-cpu" $tag }}{{- else if eq $variant "cuda" -}}{{ printf "%s-cuda" $tag }}{{- else -}}{{ $tag }}{{- end -}}
{{- end -}}

{{/*
The container image reference shared by the Ray head and the RayJob submitter.
*/}}
{{- define "modelship.image" -}}
{{- printf "%s:%s" .Values.image.repository (include "modelship.imageTag" (dict "root" .)) -}}
{{- end -}}

{{/*
Name of the ConfigMap holding models.yaml (existing or chart-templated).
*/}}
{{- define "modelship.configMapName" -}}
{{- if .Values.models.existingConfigMap -}}
{{- .Values.models.existingConfigMap -}}
{{- else -}}
{{- printf "%s-models" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret holding the HF token / API keys (existing or templated).
*/}}
{{- define "modelship.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Name of the cache PVC (existing or chart-templated).
*/}}
{{- define "modelship.cacheClaimName" -}}
{{- if .Values.cache.existingClaim -}}
{{- .Values.cache.existingClaim -}}
{{- else -}}
{{- printf "%s-cache" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
KubeRay names the cluster's head Service "<raycluster-name>-head-svc"; the
RayCluster object itself is named by modelship.fullname.
*/}}
{{- define "modelship.headServiceName" -}}
{{- printf "%s-head-svc" (include "modelship.fullname" .) -}}
{{- end -}}

{{/*
envFrom for the HF token / API keys Secret. optional:true so pods start fine
when no Secret was created (e.g. all-ungated models, no auth).
*/}}
{{- define "modelship.envFrom" -}}
- secretRef:
    name: {{ include "modelship.secretName" . }}
    optional: true
{{- end -}}

{{/*
Name of the Secret holding the Redis password (existing or the chart's own).
*/}}
{{- define "modelship.redisSecretName" -}}
{{- .Values.redis.existingSecret | default (include "modelship.secretName" .) -}}
{{- end -}}

{{/*
Full rayStartParams for a Ray node: chart-managed defaults plus the caller's
overrides (overrides win). metrics-export-port is pinned to metrics.port on every
node so Ray's Prometheus endpoint matches the `metrics` containerPort + PodMonitor
(unset, ray start binds a random port and scrapes fail). The head additionally
runs CPU-only (num-gpus 0) and binds the dashboard on all interfaces; workers keep
num-gpus caller-controlled (GPU groups omit it to autodetect).
Call with (dict "root" $ "isHead" <bool> "params" <rayStartParams>).
*/}}
{{- define "modelship.rayStartParams" -}}
{{- $defaults := dict "metrics-export-port" (.root.Values.metrics.port | toString) -}}
{{- if .isHead -}}
{{- $_ := set $defaults "num-gpus" "0" -}}
{{- $_ := set $defaults "dashboard-host" "0.0.0.0" -}}
{{- end -}}
{{- merge (deepCopy (.params | default dict)) $defaults | toYaml -}}
{{- end -}}

{{/*
Explicit env for every Ray pod (head + workers): the state-store URI the
coordinator, effective-config and /v1/responses read via get_state_store(). It MUST
be on every pod so the coordinator — scheduled on any node — agrees with the driver.

Always redis://[:$(REDIS_PASSWORD)@]<addr>/<db> (password kept in the Secret; k8s
expands $(REDIS_PASSWORD) so it never lands in the manifest/argv). The same Redis
also backs GCS fault tolerance. The chart wires an address but does not deploy Redis,
so redis.address is required — there is no durable fallback to degrade to.
*/}}
{{- define "modelship.env" -}}
{{- $addr := required "redis.address is required: modelship on k8s stores its effective config, routing registry and /v1/responses conversations in Redis. Point redis.address at a Redis instance (see the chart README)." .Values.redis.address }}
{{- if or .Values.redis.password .Values.redis.existingSecret }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "modelship.redisSecretName" . }}
      key: {{ .Values.redis.passwordKey }}
- name: MSHIP_STATE_STORE
  value: "redis://:$(REDIS_PASSWORD)@{{ $addr }}/{{ .Values.redis.db }}"
{{- else }}
- name: MSHIP_STATE_STORE
  value: "redis://{{ $addr }}/{{ .Values.redis.db }}"
{{- end }}
{{- end -}}

{{/*
Volumes shared by every Ray pod (head + workers): an in-memory /dev/shm for
vLLM/NCCL, and the model-weight cache PVC.
*/}}
{{- define "modelship.volumes" -}}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.shm.sizeLimit }}
{{- if .Values.cache.enabled }}
- name: cache
  persistentVolumeClaim:
    claimName: {{ include "modelship.cacheClaimName" . }}
{{- end }}
{{- end -}}

{{/*
Matching volumeMounts for the volumes above.
*/}}
{{- define "modelship.volumeMounts" -}}
- name: dshm
  mountPath: /dev/shm
{{- if .Values.cache.enabled }}
- name: cache
  mountPath: {{ .Values.cache.mountPath }}
{{- end }}
{{- end -}}
