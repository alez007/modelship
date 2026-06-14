#!/bin/bash
set -e

usage() {
    echo "Usage: start_ray.sh [--num-cpus <n>] [--num-gpus <n>] [--enable-metrics <true|false>]"
    exit 1
}

ENABLE_METRICS="true"
NUM_CPUS=""
NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-cpus) NUM_CPUS="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --enable-metrics) ENABLE_METRICS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

RAY_FLAGS=(--head --disable-usage-stats)
if [ "${MSHIP_RAY_DASHBOARD}" = "true" ]; then
    RAY_FLAGS+=(--include-dashboard=true --dashboard-host=0.0.0.0)
else
    RAY_FLAGS+=(--include-dashboard=false)
fi

if [ -n "${NUM_CPUS}" ]; then
    RAY_FLAGS+=(--num-cpus="${NUM_CPUS}")
fi

if [ -n "${NUM_GPUS}" ]; then
    RAY_FLAGS+=(--num-gpus="${NUM_GPUS}")
fi

if [ "${ENABLE_METRICS}" = "true" ]; then
    RAY_FLAGS+=(--metrics-export-port="${RAY_METRICS_EXPORT_PORT:-8079}")
fi

ray start "${RAY_FLAGS[@]}"

if ! ray status; then
    echo "ray cluster failed to start"
    exit 1
fi
