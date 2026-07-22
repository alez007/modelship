#!/bin/bash
set -e

# Use the environment variables set during build, or default to 1000
TARGET_UID=${MSHIP_UID:-1000}
TARGET_GID=${MSHIP_GID:-1000}

# The prod image's ENTRYPOINT bakes in a leading "--serve" marker (see Dockerfile) so that
# `docker run <image> --address ... --no-metrics` (trailing args replace CMD, never
# ENTRYPOINT) always lands here as flags for mship_deploy.py, without needing to retype
# `uv run --no-sync mship_deploy.py` every time. The dev image's ENTRYPOINT omits the
# marker, so it keeps today's plain passthrough (e.g. dropping into bash by default).
if [ "$1" = "--serve" ]; then
    shift
    set -- uv run --no-sync mship_deploy.py "$@"
fi

# When started as root (the standalone `docker run` path), fix up ownership for
# any root-owned bind mount and drop privileges to the unprivileged user before
# exec'ing the command.
#
# Under Kubernetes (incl. KubeRay) the pod instead sets
# `securityContext.runAsUser`, so this script may already be running
# unprivileged. A non-root user cannot chown files it doesn't own — the chown
# would fail under `set -e` and crash the container — and there is nothing to
# drop to. In that case skip straight to the command. (KubeRay also overrides
# the container command, bypassing this ENTRYPOINT entirely for Ray pods; this
# branch covers any pod that keeps it, e.g. the deploy Job.)
if [ "$(id -u)" = "0" ]; then
    # Fix permissions for the cache directory (may be a root-owned bind mount).
    # `chown -R` walks the whole weight cache, which is very slow on NFS/EFS, so
    # only do it when the directory isn't already owned by the target user — i.e.
    # the first run / a freshly-mounted root-owned volume. Restarts skip the walk.
    if [ -d "/.cache" ] && [ "$(stat -c '%u:%g' /.cache)" != "$TARGET_UID:$TARGET_GID" ]; then
        chown -R "$TARGET_UID:$TARGET_GID" /.cache
    fi
    # Also ensure the workspace has the right permissions.
    chown "$TARGET_UID:$TARGET_GID" /modelship
    # Drop privileges and execute the main command (gosu takes UID:GID).
    exec gosu "$TARGET_UID:$TARGET_GID" "$@"
fi

# Already unprivileged (k8s securityContext): run the command as-is.
exec "$@"
