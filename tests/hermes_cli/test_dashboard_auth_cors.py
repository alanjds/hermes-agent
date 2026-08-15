"""Pure-function tests for ``hermes_cli.dashboard_auth.cors``.

Mirrors ``test_dashboard_auth_prefix.py``'s style for the sibling
``public_url`` helpers, but this is a security allowlist rather than a UX
convenience: a malformed entry must raise, not silently drop.
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth.cors import (
    InvalidOriginError,
    resolve_allowed_origins,
    validate_origin,
)


# ---------------------------------------------------------------------------
# validate_origin
# ---------------------------------------------------------------------------


class TestValidateOriginAccepts:
    def test_plain_http_origin(self):
        assert validate_origin("http://192.168.1.50:4174") == "http://192.168.1.50:4174"

    def test_plain_https_origin(self):
        assert validate_origin("https://example.com") == "https://example.com"

    def test_hostname_is_lowercased(self):
        assert validate_origin("http://EXAMPLE.com:4174") == "http://example.com:4174"

    def test_default_http_port_stripped(self):
        assert validate_origin("http://example.com:80") == "http://example.com"

    def test_default_https_port_stripped(self):
        assert validate_origin("https://example.com:443") == "https://example.com"

    def test_non_default_port_kept(self):
        assert validate_origin("https://example.com:8443") == "https://example.com:8443"

    def test_trailing_slash_path_accepted(self):
        # An Origin header itself never carries a path, but an operator
        # pasting a full URL with a trailing slash is a common typo shape —
        # accept "/" specifically (not any other path).
        assert validate_origin("http://example.com/") == "http://example.com"

    def test_leading_trailing_whitespace_stripped(self):
        assert validate_origin("  http://example.com  ") == "http://example.com"


class TestValidateOriginRejects:
    @pytest.mark.parametrize("bad", [
        "",
        "   ",
    ])
    def test_empty(self, bad):
        with pytest.raises(InvalidOriginError):
            validate_origin(bad)

    @pytest.mark.parametrize("bad", [
        "*",
        "http://*.example.com",
        "http://example.com:*",
    ])
    def test_wildcard(self, bad):
        with pytest.raises(InvalidOriginError, match="wildcard"):
            validate_origin(bad)

    @pytest.mark.parametrize("bad", [
        "example.com",  # missing scheme
        "//example.com",
        "ftp://example.com",
        "ws://example.com",
        "javascript:alert(1)",
    ])
    def test_bad_scheme(self, bad):
        with pytest.raises(InvalidOriginError):
            validate_origin(bad)

    def test_missing_host(self):
        with pytest.raises(InvalidOriginError, match="host"):
            validate_origin("https://")

    def test_userinfo_rejected(self):
        with pytest.raises(InvalidOriginError, match="userinfo"):
            validate_origin("https://user:pass@example.com")

    @pytest.mark.parametrize("bad", [
        "http://example.com/path",
        "http://example.com/api",
    ])
    def test_path_rejected(self, bad):
        with pytest.raises(InvalidOriginError, match="path"):
            validate_origin(bad)

    def test_query_rejected(self):
        with pytest.raises(InvalidOriginError, match="query"):
            validate_origin("http://example.com/?x=1")

    def test_fragment_rejected(self):
        with pytest.raises(InvalidOriginError, match="query"):
            validate_origin("http://example.com/#frag")

    @pytest.mark.parametrize("bad", [
        'http://example.com/"injected',
        "http://example.com<script>",
        "http://example.com\nhttp://evil",
        "http://exa mple.com",
    ])
    def test_reject_chars(self, bad):
        with pytest.raises(InvalidOriginError):
            validate_origin(bad)


# ---------------------------------------------------------------------------
# resolve_allowed_origins
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_config(monkeypatch):
    """Replace ``hermes_cli.config.load_config`` with a stub returning the
    given ``allowed_origins`` list. Pass ``None`` for no config-side value.
    """

    def _set(allowed_origins) -> None:
        cfg = {}
        if allowed_origins is not None:
            cfg = {"dashboard": {"allowed_origins": allowed_origins}}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    return _set


class TestResolveAllowedOrigins:
    def test_empty_by_default(self, patch_config, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(None)
        assert resolve_allowed_origins() == ()

    def test_config_yaml_list_used_when_env_unset(self, patch_config, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(["http://192.168.1.50:4174"])
        assert resolve_allowed_origins() == ("http://192.168.1.50:4174",)

    def test_env_overrides_config_yaml(self, patch_config, monkeypatch):
        """Precedence pin — env wins over config.yaml, matching every other
        dashboard.* env/config pair (e.g. public_url)."""
        monkeypatch.setenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", "http://from-env.example:4174")
        patch_config(["http://from-config.example:4174"])
        assert resolve_allowed_origins() == ("http://from-env.example:4174",)

    def test_env_is_comma_separated(self, patch_config, monkeypatch):
        monkeypatch.setenv(
            "HERMES_DASHBOARD_ALLOWED_ORIGINS",
            "http://a.example:1, http://b.example:2 ,http://c.example:3",
        )
        patch_config(None)
        assert resolve_allowed_origins() == (
            "http://a.example:1", "http://b.example:2", "http://c.example:3",
        )

    def test_empty_env_falls_through_to_config(self, patch_config, monkeypatch):
        """An empty env var doesn't shadow a valid config.yaml entry —
        same defensive behaviour as ``HERMES_DASHBOARD_PUBLIC_URL``."""
        monkeypatch.setenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", "")
        patch_config(["http://from-config.example:4174"])
        assert resolve_allowed_origins() == ("http://from-config.example:4174",)

    def test_dedups_and_normalises(self, patch_config, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config([
            "http://example.com:4174",
            "HTTP://EXAMPLE.COM:4174",
            "http://example.com:4174/",
        ])
        assert resolve_allowed_origins() == ("http://example.com:4174",)

    def test_malformed_config_entry_raises(self, patch_config, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config(["not-a-url"])
        with pytest.raises(InvalidOriginError):
            resolve_allowed_origins()

    def test_malformed_env_entry_raises(self, patch_config, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", "http://ok.example, *")
        patch_config(None)
        with pytest.raises(InvalidOriginError, match="wildcard"):
            resolve_allowed_origins()

    def test_non_list_config_value_treated_as_empty(self, patch_config, monkeypatch):
        """A malformed config.yaml (e.g. a string instead of a list)
        shouldn't crash the resolver — treat as no origins configured."""
        monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_ORIGINS", raising=False)
        patch_config("http://oops-not-a-list.example")
        assert resolve_allowed_origins() == ()
