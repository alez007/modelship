#!/usr/bin/env bash
#
# easy-run.sh — zero-config launcher for Modelship.
#
# Detects available NVIDIA GPU VRAM, picks a matching preset config from
# config/examples/gpu-{8,16,24}gb.yaml, copies it to config/models.yaml,
# and starts the docker compose stack.
#
# Usage:
#   ./easy-run.sh                 # auto-detect
#   ./easy-run.sh --preset 16gb   # force a specific preset (8gb|16gb|24gb)
#   ./easy-run.sh --dry-run       # print plan and exit
#   ./easy-run.sh --force         # overwrite config/models.yaml if it exists
#
# Env:
#   HF_TOKEN    HuggingFace token for gated models (optional for the presets).

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly EXAMPLES_DIR="${SCRIPT_DIR}/config/examples"
readonly TARGET_CONFIG="${SCRIPT_DIR}/config/models.yaml"
readonly CACHE_DIR="${SCRIPT_DIR}/models-cache"

PRESET=""
DRY_RUN=0
FORCE=0

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==> WARN:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

usage() {
    sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --preset)  PRESET="${2:-}"; shift 2 ;;
            --dry-run) DRY_RUN=1; shift ;;
            --force)   FORCE=1; shift ;;
            -h|--help) usage 0 ;;
            *)         err "Unknown argument: $1"; usage 1 ;;
        esac
    done

    if [[ -n "$PRESET" && ! "$PRESET" =~ ^(8gb|16gb|24gb)$ ]]; then
        die "--preset must be one of: 8gb, 16gb, 24gb (got: $PRESET)"
    fi
}

check_requirements() {
    command -v docker >/dev/null || die "docker is not installed or not on PATH."
    if ! docker compose version >/dev/null 2>&1; then
        die "docker compose v2 is required. Install the Compose plugin: https://docs.docker.com/compose/install/"
    fi
}

# Pick preset based on the smallest GPU's VRAM (MiB). Round down to the tier.
# < 8 GiB   -> not supported (suggest CPU example)
# 8..15 GiB -> 8gb
# 16..23    -> 16gb
# >= 24     -> 24gb
detect_preset() {
    if ! command -v nvidia-smi >/dev/null; then
        die "nvidia-smi not found. Install the NVIDIA driver + Container Toolkit, or pass --preset <size> to override."
    fi

    local mem_mib
    mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NR==1 || $1 < min { min=$1 } END { print min+0 }')

    if [[ -z "$mem_mib" || "$mem_mib" -le 0 ]]; then
        die "Could not read GPU memory from nvidia-smi. Pass --preset <size> to override."
    fi

    # nvidia-smi reports slightly less than nameplate VRAM (e.g. 8188 MiB for an 8 GB card).
    # Use GiB thresholds with a small tolerance.
    local mem_gib=$(( mem_mib / 1024 ))
    log "Detected GPU VRAM: ${mem_mib} MiB (~${mem_gib} GiB)"

    if   (( mem_mib >= 23000 )); then PRESET="24gb"
    elif (( mem_mib >= 15000 )); then PRESET="16gb"
    elif (( mem_mib >=  7500 )); then PRESET="8gb"
    else
        die "GPU has ${mem_gib} GiB VRAM — below the 8 GB minimum for the GPU presets. Try the CPU example: config/examples/mini-pc.yaml"
    fi
}

install_config() {
    local src="${EXAMPLES_DIR}/gpu-${PRESET}.yaml"
    [[ -f "$src" ]] || die "Preset file not found: $src"

    if [[ -f "$TARGET_CONFIG" && "$FORCE" -ne 1 ]]; then
        warn "config/models.yaml already exists — keeping it. Use --force to overwrite."
        return
    fi

    mkdir -p "$(dirname "$TARGET_CONFIG")"
    cp "$src" "$TARGET_CONFIG"
    log "Installed preset: gpu-${PRESET}.yaml -> config/models.yaml"
}

cache_populated() {
    [[ -d "$CACHE_DIR/hub" ]] && return 0
    compgen -G "$CACHE_DIR/models--*" >/dev/null 2>&1 && return 0
    return 1
}

preset_download_estimate() {
    case "$PRESET" in
        8gb)  echo "~6 GB download, 5–15 min" ;;
        16gb) echo "~20 GB download, 10–25 min" ;;
        24gb) echo "~30 GB download, 15–40 min" ;;
        *)    echo "several GB download, 10–30 min" ;;
    esac
}

start_stack() {
    mkdir -p "$CACHE_DIR"

    log "Starting Modelship (API :8000, metrics :8079, Ray dashboard :8265)..."

    if cache_populated; then
        log "Cache at ./models-cache populated — subsequent-run startup is typically 1–2 min."
    else
        printf '\n'
        warn "FIRST RUN — expect a long startup."
        warn "  Preset ${PRESET}: $(preset_download_estimate)."
        warn "  vLLM also compiles CUDA kernels on first engine init."
        warn "  Weights persist in ./models-cache for all future runs."
        printf '\n'
        log "Monitor readiness + see per-model progress:"
        log "    ./easy-test.sh --timeout 1800"
        log "Or tail raw logs:"
        log "    docker compose logs -f modelship"
        printf '\n'
    fi
    log "Stop with: docker compose down"

    # -d so users get their shell back; logs are one command away.
    exec docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
}

main() {
    parse_args "$@"
    check_requirements

    if [[ -z "$PRESET" ]]; then
        detect_preset
    else
        log "Using forced preset: ${PRESET}"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run — would install config/examples/gpu-${PRESET}.yaml and run 'docker compose up -d'."
        exit 0
    fi

    install_config
    start_stack
}

main "$@"
