"""Tests for the dashboard's CORS policy: loopback-only by default, plus an
opt-in exact-origin allowlist for browser-hosted clients on a different
origin than the dashboard (e.g. ``apps/desktop``'s browser-fallback mode,
served statically on its own port, talking to the dashboard over
``fetch``/WebSocket).

Two layers are covered:

  1. ``_dashboard_cors_kwargs`` / ``CORSMiddleware`` response headers —
     via a throwaway ``FastAPI()`` + ``TestClient``, because Starlette
     locks ``add_middleware()`` after the first request, so the real
     ``web_server.app`` singleton can't be reconfigured mid-test-session.
  2. ``start_server()`` — the loopback precondition, the malformed-origin
     fail-closed path, and the allowed-origins log line.
"""
from __future__ import annotations

import logging

import pytest

# Same xdist group as the other dashboard-auth test files — none of these
# tests mutate web_server.app.state, but test_web_server.py's other files
# do, and start_server() here touches the real app.state.auth_required /
# allowed_origins, so keep it out of any race with those.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.cors import InvalidOriginError


# ---------------------------------------------------------------------------
# _dashboard_cors_kwargs / CORSMiddleware response headers
# ---------------------------------------------------------------------------


def _throwaway_app(extra_origins=()):
    app = FastAPI()
    app.add_middleware(CORSMiddleware, **web_server._dashboard_cors_kwargs(tuple(extra_origins)))

    @app.get("/api/probe")
    def _probe():
        return {"ok": True}

    return TestClient(app)


class TestCorsHeaders:
    def test_loopback_origin_always_allowed(self):
        client = _throwaway_app()
        r = client.get("/api/probe", headers={"Origin": "http://127.0.0.1:9119"})
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:9119"
        assert r.headers.get("access-control-allow-credentials") == "true"

    def test_localhost_origin_always_allowed(self):
        client = _throwaway_app()
        r = client.get("/api/probe", headers={"Origin": "http://localhost:4174"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:4174"

    def test_unconfigured_extra_origin_rejected(self):
        client = _throwaway_app()
        r = client.get("/api/probe", headers={"Origin": "http://192.168.1.50:4174"})
        assert "access-control-allow-origin" not in r.headers

    def test_configured_extra_origin_allowed_alongside_loopback(self):
        client = _throwaway_app(extra_origins=["http://192.168.1.50:4174"])
        r_extra = client.get("/api/probe", headers={"Origin": "http://192.168.1.50:4174"})
        assert r_extra.headers.get("access-control-allow-origin") == "http://192.168.1.50:4174"
        r_loopback = client.get("/api/probe", headers={"Origin": "http://127.0.0.1:9119"})
        assert r_loopback.headers.get("access-control-allow-origin") == "http://127.0.0.1:9119"

    def test_configured_extra_origin_never_echoes_as_wildcard(self):
        client = _throwaway_app(extra_origins=["http://192.168.1.50:4174"])
        r = client.get("/api/probe", headers={"Origin": "http://192.168.1.50:4174"})
        assert r.headers.get("access-control-allow-origin") != "*"

    def test_other_extra_origin_still_rejected(self):
        """Configuring one extra origin doesn't open the door to others."""
        client = _throwaway_app(extra_origins=["http://192.168.1.50:4174"])
        r = client.get("/api/probe", headers={"Origin": "http://10.0.0.9:4174"})
        assert "access-control-allow-origin" not in r.headers

    def test_preflight_options_request(self):
        client = _throwaway_app(extra_origins=["http://192.168.1.50:4174"])
        r = client.options(
            "/api/probe",
            headers={
                "Origin": "http://192.168.1.50:4174",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Hermes-Session-Token",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://192.168.1.50:4174"
        assert r.headers.get("access-control-allow-credentials") == "true"


# ---------------------------------------------------------------------------
# start_server(): loopback precondition, fail-closed, logging
# ---------------------------------------------------------------------------


def _stub_uvicorn_run(monkeypatch):
    """Same fake as test_dashboard_auth_gate.py's ``_stub_uvicorn_run`` —
    duplicated locally to avoid a cross-file import dependency for a small
    fixture. Replaces uvicorn.Config/Server with no-op fakes so
    start_server() returns immediately."""
    import contextlib
    import uvicorn

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            pass

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *a, **kw):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", lambda config: _FakeServer())


@pytest.fixture
def patch_config(monkeypatch):
    def _set(allowed_origins) -> None:
        cfg = {}
        if allowed_origins is not None:
            cfg = {"dashboard": {"allowed_origins": allowed_origins}}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    return _set


@pytest.fixture
def restore_app_state():
    """Save/restore the app.state attributes these tests mutate, so a
    failure here doesn't leak into other test files in the same worker."""
    prev = {
        k: getattr(web_server.app.state, k, None)
        for k in ("auth_required", "bound_host", "allowed_origins")
    }
    yield
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


class TestStartServerAllowedOrigins:
    def test_loopback_with_configured_origins_fails_closed(
        self, monkeypatch, patch_config, restore_app_state
    ):
        """Setting allowed_origins while still bound to loopback can't do
        anything useful (loopback is unreachable from another origin's
        browser regardless of CORS) and almost always means a config
        copied from a gated deploy onto a loopback dev box — refuse to
        start with a clear error rather than silently running inert."""
        _stub_uvicorn_run(monkeypatch)
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(["http://192.168.1.50:4174"])

        with pytest.raises(SystemExit, match="allowed_origins") as exc_info:
            web_server.start_server(
                host="127.0.0.1", port=9119, open_browser=False, allow_public=False,
            )
        assert "loopback" in str(exc_info.value)

    def test_gated_mode_with_configured_origins_succeeds(
        self, monkeypatch, patch_config, restore_app_state, caplog
    ):
        from hermes_cli.dashboard_auth import clear_providers, register_provider
        from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

        clear_providers()
        register_provider(StubAuthProvider())
        try:
            _stub_uvicorn_run(monkeypatch)
            monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
            patch_config(["http://192.168.1.50:4174"])

            with caplog.at_level(logging.WARNING, logger=web_server._log.name):
                web_server.start_server(
                    host="192.168.1.50", port=9119, open_browser=False, allow_public=False,
                )

            assert web_server.app.state.allowed_origins == ("http://192.168.1.50:4174",)
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any(
                "192.168.1.50:4174" in m and "trusting" in m for m in warnings
            ), f"expected an allowed-origins log line, got: {warnings!r}"
        finally:
            clear_providers()

    def test_no_configured_origins_emits_no_warning_and_empty_tuple(
        self, monkeypatch, patch_config, restore_app_state, caplog
    ):
        _stub_uvicorn_run(monkeypatch)
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(None)

        with caplog.at_level(logging.WARNING, logger=web_server._log.name):
            web_server.start_server(
                host="127.0.0.1", port=9119, open_browser=False, allow_public=False,
            )

        assert web_server.app.state.allowed_origins == ()
        assert not [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "allowed_origins" in r.getMessage()
        ]

    def test_malformed_origin_fails_closed(
        self, monkeypatch, patch_config, restore_app_state
    ):
        _stub_uvicorn_run(monkeypatch)
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(["not-a-valid-origin"])

        with pytest.raises(SystemExit, match="Invalid dashboard.allowed_origins"):
            web_server.start_server(
                host="127.0.0.1", port=9119, open_browser=False, allow_public=False,
            )


class TestCorsWrapsAuthRejections:
    """Regression guard: CORSMiddleware must be the OUTERMOST middleware on
    the real app, wrapping the auth-gate middlewares — not nested inside
    them.

    Concretely reproduces the bug hit when testing apps/desktop's
    browser-fallback mode against a loopback dashboard from a different
    origin/port: with CORSMiddleware registered before (i.e. innermost of)
    auth_middleware/the OAuth gate/host_header_middleware, an unauthenticated
    preflight ``OPTIONS`` request got 401'd by auth_middleware before ever
    reaching CORSMiddleware — so the browser's actual request (e.g.
    ``fetch`` with the ``X-Hermes-Session-Token`` header, which forces a
    preflight) never went out at all, surfacing as a bare "NetworkError".
    Same root cause made every auth-rejection response (401/400) come back
    without ``Access-Control-Allow-Origin``, which a browser also treats as
    an opaque network failure rather than a readable error body.
    """

    def test_preflight_not_blocked_by_auth_middleware(
        self, restore_app_state
    ):
        web_server.app.state.bound_host = None
        web_server.app.state.auth_required = False
        client = TestClient(web_server.app)

        r = client.options(
            "/api/config",
            headers={
                "Origin": "http://127.0.0.1:5174",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-hermes-session-token",
            },
        )
        assert r.status_code == 200, (
            f"preflight must be answered by CORSMiddleware, not 401'd by "
            f"an auth gate: got {r.status_code}: {r.text}"
        )
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5174"

    def test_unauthenticated_rejection_still_carries_cors_headers(
        self, restore_app_state
    ):
        web_server.app.state.bound_host = None
        web_server.app.state.auth_required = False
        client = TestClient(web_server.app)

        r = client.get("/api/config", headers={"Origin": "http://127.0.0.1:5174"})
        assert r.status_code == 401
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5174", (
            "an auth-rejection response must still be CORS-readable by the "
            "caller, or a browser reports it as a NetworkError instead of "
            "a 401 the SPA can act on"
        )

    def test_authenticated_request_carries_cors_headers(self, restore_app_state):
        web_server.app.state.bound_host = None
        web_server.app.state.auth_required = False
        client = TestClient(web_server.app)

        r = client.get(
            "/api/config",
            headers={
                "Origin": "http://127.0.0.1:5174",
                "X-Hermes-Session-Token": web_server._SESSION_TOKEN,
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5174"


def test_resolve_allowed_origins_reexported_for_module_import_time_check():
    """Sanity check that InvalidOriginError is importable the way
    web_server.py itself imports it (regression guard for the import-time
    SystemExit wrapping at module load)."""
    assert issubclass(InvalidOriginError, ValueError)
