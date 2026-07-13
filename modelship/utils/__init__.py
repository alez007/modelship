import logging
import os
import random
import string
import uuid
from collections.abc import Iterable
from typing import Any

import requests

from modelship.infer.infer_config import RawRequestProxy

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


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def rand_suffix(length: int = 5) -> str:
    return "".join(random.choices(_RAND_CHARS, k=length))


def base_request_id(raw_request: RawRequestProxy | None = None) -> str:
    """Return the request ID from a RawRequestProxy, or generate a new one."""
    if raw_request is not None and raw_request.request_id is not None:
        return raw_request.request_id
    return random_uuid()


def download(url: str, file_path: str, overwrite: bool = False):
    """Download ``url`` to ``file_path``, skipping if it already exists.

    Streams to a per-call unique temp file and atomically renames it into place
    only after the transfer completes and matches the advertised size, so a
    killed process or an early-EOF stream never leaves a truncated file at
    ``file_path``.
    """
    if not overwrite and os.path.isfile(file_path):
        return

    tmp_path = f"{file_path}.{random_uuid()}.tmp"
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length") or 0)
            written = 0
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        written += f.write(chunk)
        if expected and written != expected:
            raise OSError(f"incomplete download from {url}: got {written} of {expected} bytes")
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
