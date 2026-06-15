#!/bin/bash
set -e

# Use the environment variables set during build, or default to 1000
TARGET_UID=${MSHIP_UID:-1000}
TARGET_GID=${MSHIP_GID:-1000}

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
    if [ -d "/.cache" ]; then
        chown -R "$TARGET_UID:$TARGET_GID" /.cache
    fi
    # Also ensure the workspace has the right permissions.
    chown "$TARGET_UID:$TARGET_GID" /modelship
    # Drop privileges and execute the main command (gosu takes UID:GID).
    exec gosu "$TARGET_UID:$TARGET_GID" "$@"
fi

# Already unprivileged (k8s securityContext): run the command as-is.
exec "$@"
