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
The container image reference shared by Ray pods and the deploy Job.
*/}}
{{- define "modelship.image" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
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
Volumes shared by every Ray pod and the deploy Job: an in-memory /dev/shm for
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
