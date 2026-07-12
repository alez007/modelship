"""Tests for API key authentication middleware."""

import hashlib
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from modelship.openai.auth import (
    UNSCOPED_IDENTITY,
    ApiKeyMiddleware,
    get_api_keys,
    get_trusted_identity_header,
    identity_key,
    identity_tier,
    resolve_identity,
)


def _make_request(headers: dict[str, str] | None = None) -> Request:
    """Build a bare Request carrying only the given headers, for identity_key/identity_tier tests."""
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return Request(scope)


def _make_app(api_keys: set[str]) -> FastAPI:
    """Build a minimal FastAPI app with the auth middleware for testing."""
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_keys=api_keys)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return JSONResponse({"id": "test"})

    return app


VALID_KEY = "sk-test-key-123"
OTHER_KEY = "sk-other-key-456"
KEYS = {VALID_KEY, OTHER_KEY}


class TestApiKeyMiddleware:
    def test_valid_key_allows_request(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {VALID_KEY}"})
        assert resp.status_code == 200

    def test_other_valid_key_allows_request(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {OTHER_KEY}"})
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]["message"]

    def test_empty_bearer_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]["message"]

    def test_invalid_key_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]["message"]

    def test_non_bearer_auth_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Basic {VALID_KEY}"})
        assert resp.status_code == 401

    def test_health_endpoint_bypasses_auth(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_post_endpoint_requires_auth(self):
        client = TestClient(_make_app(KEYS))
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 401

    def test_post_endpoint_with_valid_key(self):
        client = TestClient(_make_app(KEYS))
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
        assert resp.status_code == 200

    def test_error_response_format(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models")
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"
        assert body["error"]["code"] == 401


class TestGetApiKeys:
    def test_returns_keys_from_env(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a,sk-b,sk-c"}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b", "sk-c"}

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": " sk-a , sk-b "}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b"}

    def test_empty_env_returns_empty_set(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": ""}):
            keys = get_api_keys()
        assert keys == set()

    def test_unset_env_returns_empty_set(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = get_api_keys()
        assert keys == set()

    def test_ignores_empty_entries(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a,,,,sk-b,"}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b"}


class TestIdentityKey:
    def test_trusted_header_used_raw(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id", "MSHIP_API_KEYS": ""}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "customer-42"})
            assert identity_key(request) == "customer-42"

    def test_trusted_header_stripped(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "  customer-42  "})
            assert identity_key(request) == "customer-42"

    def test_header_configured_but_absent_falls_back_to_matched_key(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id", "MSHIP_API_KEYS": "sk-a"}
        with patch.dict(os.environ, env):
            request = _make_request({"Authorization": "Bearer sk-a"})
            assert identity_key(request) == hashlib.sha256(b"sk-a").hexdigest()

    def test_header_configured_but_empty_falls_back(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id", "MSHIP_API_KEYS": "sk-a"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "   ", "Authorization": "Bearer sk-a"})
            assert identity_key(request) == hashlib.sha256(b"sk-a").hexdigest()

    def test_header_wins_over_matched_key(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id", "MSHIP_API_KEYS": "sk-a"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "customer-42", "Authorization": "Bearer sk-a"})
            assert identity_key(request) == "customer-42"

    def test_matched_key_hashed_when_no_header_configured(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a,sk-b"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request({"Authorization": "Bearer sk-a"})
            assert identity_key(request) == hashlib.sha256(b"sk-a").hexdigest()

    def test_distinct_keys_yield_distinct_hashes(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a,sk-b"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            id_a = identity_key(_make_request({"Authorization": "Bearer sk-a"}))
            id_b = identity_key(_make_request({"Authorization": "Bearer sk-b"}))
        assert id_a != id_b

    def test_no_header_no_key_returns_unscoped_sentinel(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": ""}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request()
            result = identity_key(request)
        assert result == UNSCOPED_IDENTITY
        # Must never collide with a real sha256 hex digest (64 lowercase hex chars).
        assert len(result) != 64 or not all(c in "0123456789abcdef" for c in result)

    def test_unsafe_header_value_falls_back_to_hash(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            unsafe = "../../etc/passwd"
            request = _make_request({"X-Consumer-Id": unsafe})
            result = identity_key(request)
        assert result == hashlib.sha256(unsafe.encode()).hexdigest()
        assert "/" not in result

    def test_overlong_header_value_falls_back_to_hash(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            overlong = "a" * 200
            request = _make_request({"X-Consumer-Id": overlong})
            result = identity_key(request)
        assert result == hashlib.sha256(overlong.encode()).hexdigest()

    def test_dot_dot_header_value_falls_back_to_hash(self):
        """ ".." matches the charset (dots are allowed mid-value, e.g. "svc.billing") but as
        a whole segment would traverse a future file-based state store keyed by this value."""
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": ".."})
            result = identity_key(request)
        assert result == hashlib.sha256(b"..").hexdigest()

    def test_single_dot_header_value_falls_back_to_hash(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "."})
            result = identity_key(request)
        assert result == hashlib.sha256(b".").hexdigest()


class TestIdentityTier:
    def test_header_tier(self):
        env = {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-Consumer-Id"}
        with patch.dict(os.environ, env):
            request = _make_request({"X-Consumer-Id": "customer-42"})
            assert identity_tier(request) == "header"

    def test_api_key_tier(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request({"Authorization": "Bearer sk-a"})
            assert identity_tier(request) == "api_key"

    def test_unscoped_tier(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": ""}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request()
            assert identity_tier(request) == "unscoped"


class TestEnvCaching:
    """get_api_keys()/get_trusted_identity_header() cache on the raw env string, so a
    changed env value must still be picked up (only an unchanged raw string should hit cache)."""

    def test_get_api_keys_reflects_env_change(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a"}, clear=False):
            assert get_api_keys() == {"sk-a"}
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-b"}, clear=False):
            assert get_api_keys() == {"sk-b"}

    def test_get_trusted_identity_header_reflects_env_change(self):
        with patch.dict(os.environ, {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-A"}, clear=False):
            assert get_trusted_identity_header() == "X-A"
        with patch.dict(os.environ, {"MSHIP_TRUSTED_IDENTITY_HEADER": "X-B"}, clear=False):
            assert get_trusted_identity_header() == "X-B"


class TestResolveIdentityCaching:
    """resolve_identity() caches its result on request.state so a second call for the
    same request is free and stable even if env vars change in between."""

    def test_result_is_cached_on_request_state(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request({"Authorization": "Bearer sk-a"})
            first = resolve_identity(request)
            os.environ["MSHIP_API_KEYS"] = "sk-b"
            second = resolve_identity(request)
        assert first == second == (hashlib.sha256(b"sk-a").hexdigest(), "api_key")

    def test_identity_key_and_identity_tier_share_one_resolution(self):
        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _make_request({"Authorization": "Bearer sk-a"})
            key = identity_key(request)
            os.environ["MSHIP_API_KEYS"] = "sk-b"
            tier = identity_tier(request)
        assert key == hashlib.sha256(b"sk-a").hexdigest()
        assert tier == "api_key"

    def test_request_like_object_without_state_attribute(self):
        """A request-like object with no `state` attribute at all must not raise —
        caching is simply skipped, resolution still succeeds every call."""

        class _NoStateRequest:
            def __init__(self, headers: dict[str, str]) -> None:
                self.headers = headers

        with patch.dict(os.environ, {"MSHIP_API_KEYS": "sk-a"}, clear=False):
            os.environ.pop("MSHIP_TRUSTED_IDENTITY_HEADER", None)
            request = _NoStateRequest({"authorization": "Bearer sk-a"})
            assert not hasattr(request, "state")
            result = resolve_identity(request)  # type: ignore[arg-type]
        assert result == (hashlib.sha256(b"sk-a").hexdigest(), "api_key")
