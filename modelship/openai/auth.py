import hashlib
import hmac
import os
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from modelship.logging import get_logger
from modelship.metrics import AUTH_FAILURES_TOTAL

logger = get_logger("api.auth")

_PUBLIC_PATHS = {"/health"}

# Sentinel returned by identity_key() when no identity is resolvable (no trusted
# header, no matched API key). Deliberately not hash-shaped (a sha256 hex digest
# is 64 lowercase hex chars) so it can never collide with a real identity value.
# Every caller with no resolvable identity shares this one bucket.
UNSCOPED_IDENTITY = "unscoped"

# Charset a trusted-header identity value must match to be used raw (as a log
# field / state-key segment). Anything outside this — newlines, "/", control
# chars, or overlong values — falls back to a sha256 hash instead of ever
# propagating untrusted bytes into a log line or a state-store key. "." is in
_SAFE_IDENTITY_RE = re.compile(r"^(?!\.\.?$)[A-Za-z0-9_.:-]{1,128}$")


# (raw env string, parsed value) caches for get_api_keys()/get_trusted_identity_header().
# Keyed on the raw string rather than parsed once at import time so tests using
# patch.dict(os.environ, ...) still see up-to-date values with no manual cache clearing —
# the cache only pays off across the many requests within one unchanging-env process.
_api_keys_cache: tuple[str, set[str]] | None = None
_trusted_header_cache: tuple[str, str | None] | None = None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validates ``Authorization: Bearer <key>`` against a set of allowed API keys."""

    def __init__(self, app, api_keys: set[str]):
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""

        if not token:
            AUTH_FAILURES_TOTAL.inc(tags={"reason": "missing"})
            logger.warning("auth failed (missing key): %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing API key. Use Authorization: Bearer <key>.",
                        "type": "auth_error",
                        "code": 401,
                    }
                },
            )

        if _matched_api_key(token, self.api_keys) is None:
            AUTH_FAILURES_TOTAL.inc(tags={"reason": "invalid"})
            logger.warning("auth failed (invalid key): %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key.", "type": "auth_error", "code": 401}},
            )

        return await call_next(request)


def _matched_api_key(token: str, keys: set[str]) -> str | None:
    """Return the key in *keys* that constant-time-matches *token*, or None."""
    return next((key for key in keys if hmac.compare_digest(token, key)), None)


def get_api_keys() -> set[str]:
    """Read allowed API keys from the ``MSHIP_API_KEYS`` environment variable (comma-separated)."""
    global _api_keys_cache
    raw = os.environ.get("MSHIP_API_KEYS", "")
    cached = _api_keys_cache
    if cached is not None and cached[0] == raw:
        return cached[1]
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    _api_keys_cache = (raw, keys)
    return keys


def get_trusted_identity_header() -> str | None:
    """Read the trusted identity header name from ``MSHIP_TRUSTED_IDENTITY_HEADER``, if set."""
    global _trusted_header_cache
    raw = os.environ.get("MSHIP_TRUSTED_IDENTITY_HEADER", "")
    cached = _trusted_header_cache
    if cached is not None and cached[0] == raw:
        return cached[1]
    value = raw.strip() or None
    _trusted_header_cache = (raw, value)
    return value


def resolve_identity(request: Request) -> tuple[str, str]:
    """Resolve (identity_key, identity_tier) in one pass and cache the result on ``request.state``.

    Not an auth check — never rejects a request. Resolution order:

    1. A configured ``MSHIP_TRUSTED_IDENTITY_HEADER`` present on the request: the
       raw header value (sanitized) — a non-secret identifier an operator's
       credentials layer assigned, kept legible in logs/state keys. Requires that
       layer to unconditionally overwrite the header and modelship to be
       unreachable except from it (see docs/model-configuration.md). Tier: "header".
    2. The matched ``MSHIP_API_KEYS`` entry: sha256 hex (key material never
       appears in logs or keys). Tier: "api_key".
    3. Neither: ``UNSCOPED_IDENTITY`` — every such caller shares one bucket. Tier: "unscoped".

    identity_key() and identity_tier() both delegate here so the header lookup, token
    extraction, and constant-time key match happen once per request instead of twice.
    """
    state = getattr(request, "state", None)
    cached = getattr(state, "_identity", None) if state is not None else None
    if isinstance(cached, tuple):
        return cached

    header_name = get_trusted_identity_header()
    if header_name:
        value = request.headers.get(header_name, "").strip()
        if value:
            key = value if _SAFE_IDENTITY_RE.match(value) else hashlib.sha256(value.encode()).hexdigest()
            result = (key, "header")
            if state is not None:
                state._identity = result
            return result

    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    if token:
        matched = _matched_api_key(token, get_api_keys())
        if matched is not None:
            result = (hashlib.sha256(matched.encode()).hexdigest(), "api_key")
            if state is not None:
                state._identity = result
            return result

    result = (UNSCOPED_IDENTITY, "unscoped")
    if state is not None:
        state._identity = result
    return result


def identity_key(request: Request) -> str:
    """Resolve a stable per-caller identity string for log correlation and future state-keying."""
    return resolve_identity(request)[0]


def identity_tier(request: Request) -> str:
    """Return which identity_key() tier resolved for *request*: "header" / "api_key" / "unscoped".

    For logging/observability only — lets an unexpected shift to "unscoped" (e.g. a
    fronting proxy that stopped setting the trusted header) show up in logs.
    """
    return resolve_identity(request)[1]
