import logging
import os
import random
import string
from collections.abc import Iterable
from typing import Any

import requests

# Re-exported from a ray-free leaf module so importing modelship.utils (hence
# modelship.utils.cli) never pulls in `import ray` — mship_deploy needs to parse
# argv and set Ray's auth env vars before its own `import ray`. base_request_id
# is a pure re-export (alias form marks that for the linter); random_uuid is also
# used by download() below.
from modelship.utils.request_id import base_request_id as base_request_id
from modelship.utils.request_id import random_uuid

_RAND_CHARS = string.ascii_lowercase + string.digits


def drop_reserved_kwargs(
    kwargs: dict[str, Any], reserved: Iterable[str], *, logger: logging.Logger, context: str
) -> dict[str, Any]:
    """Strip keys the caller passes to ``apply_chat_template`` itself.

    User-supplied ``chat_template_kwargs`` are splatted alongside explicit
    arguments (``tokenize``, ``tools``, ``add_generation_prompt``, …); a collision
    is a duplicate-keyword ``TypeError`` (or silently flips an explicit value).
    Drop the offenders with a warning so misconfiguration surfaces instead.
    """
    reserved = set(reserved)
    dropped = sorted(k for k in kwargs if k in reserved)
    if dropped:
        logger.warning("%s: ignoring reserved chat_template_kwargs %s", context, dropped)
    return {k: v for k, v in kwargs.items() if k not in reserved}


def rand_suffix(length: int = 5) -> str:
    return "".join(random.choices(_RAND_CHARS, k=length))


def download(url: str, file_path: str, overwrite: bool = False):
    """Download ``url`` to ``file_path``, skipping if it already exists.

    Streams to a per-call unique temp file and atomically renames it into place
    only on success
    """
    if not overwrite and os.path.isfile(file_path):
        return

    tmp_path = f"{file_path}.{random_uuid()}.tmp"
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def cache_dir() -> str:
    path = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
    os.makedirs(path, exist_ok=True)
    return path


def plugins_dir() -> str:
    path = f"{cache_dir()}/plugins"
    os.makedirs(path, exist_ok=True)
    return path
