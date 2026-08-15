"""Resolution and validation for ``dashboard.allowed_origins``.

The dashboard's HTTP CORS policy and its WebSocket Origin guard are both
loopback-only by default (see ``web_server.py``'s ``CORSMiddleware`` setup
and ``_ws_host_origin_reason``). That is correct for the common case — a
browser tab pointed straight at the dashboard's own origin — but it blocks a
legitimate shape: a browser-hosted client running on a *different* origin
than the dashboard (e.g. ``apps/desktop``'s browser-fallback renderer,
served statically on its own port) making cross-origin ``fetch``/WebSocket
calls back to an already-authenticated, non-loopback-bound dashboard.

This module resolves and validates an explicit, operator-typed allowlist of
extra origins to trust. Unlike ``prefix.py``'s ``public_url`` (a UX
convenience that silently falls back on a malformed value), this is a
security allowlist: any malformed entry is a fail-closed error, not a
silent drop, so a typo can never quietly widen or narrow trust in a way the
operator didn't intend.

Deliberately narrow scope, matching ``prefix.py``'s no-FastAPI-import
convention so both HTTP and WS call sites can share it without pulling in
web-framework machinery:

- Exact origins only (``scheme://host[:port]``) — never a wildcard or
  pattern. A leaked/typo'd allowlist entry can grant at most one extra
  origin, never a whole domain or scheme.
- No path/query/fragment/userinfo — an Origin header never carries those,
  so accepting them here would just be a way to misconfigure silently.
"""
from __future__ import annotations

import os
import urllib.parse
from typing import Tuple

#: Characters that, if present, indicate a typo or header-injection attempt
#: rather than a genuine origin. Mirrors ``prefix.py``'s ``_REJECT_CHARS``.
_REJECT_CHARS = frozenset(('"', "'", "<", ">", " ", "\n", "\r", "\t"))

#: Default ports that are equivalent to "no port" for the given scheme, so
#: ``http://example.com:80`` and ``http://example.com`` normalise to the
#: same allowlist entry.
_DEFAULT_PORTS = {"http": 80, "https": 443}


class InvalidOriginError(ValueError):
    """Raised when a configured origin fails validation.

    A ``ValueError`` subclass so callers that only care about "was this a
    value error" (e.g. a generic ``except ValueError`` at a config-loading
    boundary) still catch it, while call sites that want the specific type
    can catch it directly.
    """


def validate_origin(raw: str) -> str:
    """Validate and normalise a single configured origin.

    Returns the normalised form (``scheme://host[:port]``, lowercase
    scheme/host, default port stripped, no trailing slash). Raises
    :class:`InvalidOriginError` with a human-readable reason on any of:
    empty/whitespace value, a ``*`` wildcard anywhere, a non-``http(s)``
    scheme, a missing host, userinfo (``user:pass@host``), a path other
    than ``""``/``"/"``, a query or fragment, or any reject-listed
    character (quotes, angle brackets, whitespace, control chars).

    This is a security allowlist, so validation is fail-closed: unlike
    ``prefix.py``'s ``_normalise_public_url`` (which silently discards a
    malformed value as a UX nicety), a malformed origin here must stop the
    caller rather than be quietly dropped — a misconfigured deployment
    should fail loudly, not run with a smaller-than-intended allowlist.
    """
    value = raw.strip() if raw else ""
    if not value:
        raise InvalidOriginError("origin is empty")
    if "*" in value:
        raise InvalidOriginError(f"{raw!r}: wildcards are not allowed in dashboard.allowed_origins")
    if any(c in value for c in _REJECT_CHARS):
        raise InvalidOriginError(f"{raw!r}: contains a quote, angle bracket, whitespace, or control character")

    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError as exc:
        raise InvalidOriginError(f"{raw!r}: could not be parsed as a URL ({exc})") from exc

    if parsed.scheme not in {"http", "https"}:
        raise InvalidOriginError(f"{raw!r}: must start with http:// or https://")
    if parsed.username or parsed.password:
        raise InvalidOriginError(f"{raw!r}: must not include userinfo (user:pass@host)")
    if not parsed.hostname:
        raise InvalidOriginError(f"{raw!r}: missing host")
    if parsed.path not in ("", "/"):
        raise InvalidOriginError(f"{raw!r}: must not include a path — an Origin is scheme://host[:port] only")
    if parsed.query or parsed.fragment:
        raise InvalidOriginError(f"{raw!r}: must not include a query string or fragment")

    host = parsed.hostname.lower()
    port = parsed.port
    if port is not None and port == _DEFAULT_PORTS.get(parsed.scheme):
        port = None

    return f"{parsed.scheme}://{host}" if port is None else f"{parsed.scheme}://{host}:{port}"


def _load_dashboard_section() -> dict:
    """Return the ``dashboard`` block from ``config.yaml``, or ``{}``.

    Same robustness contract as ``prefix.py``'s helper of the same name:
    tolerant of ``load_config()`` raising or the section being absent /
    non-dict.
    """
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    try:
        cfg = load_config()
    except Exception:
        return {}
    section = cfg.get("dashboard") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def resolve_allowed_origins() -> Tuple[str, ...]:
    """Resolve the configured extra-origin allowlist.

    Precedence (mirrors every other ``dashboard.*`` env/config pair):

      1. ``HERMES_DASHBOARD_ALLOWED_ORIGINS`` env var — comma-separated —
         when non-empty after strip.
      2. ``dashboard.allowed_origins`` in ``config.yaml`` — a list.
      3. Empty tuple — no extra origins, CORS/WS stay loopback-only.

    Every candidate is run through :func:`validate_origin`; the first
    failure raises :class:`InvalidOriginError` — this function does not
    catch it, so the caller (``web_server.py``, at both the CORS-middleware
    import-time call site and the ``start_server()`` re-check) is
    responsible for turning it into a clean fail-closed ``SystemExit``
    rather than an ugly import-time traceback.

    Deterministic and side-effect-free: it is called twice per process
    (once to build the CORS middleware, once in ``start_server()`` to
    apply the loopback precondition), and both calls must agree — that
    agreement is a documented invariant of this function, not an
    implementation detail.
    """
    env_raw = os.environ.get("HERMES_DASHBOARD_ALLOWED_ORIGINS", "")
    if env_raw.strip():
        candidates = [c for c in (part.strip() for part in env_raw.split(",")) if c]
    else:
        cfg_value = _load_dashboard_section().get("allowed_origins", [])
        candidates = [str(c) for c in cfg_value] if isinstance(cfg_value, list) else []

    seen: dict = {}
    for candidate in candidates:
        normalised = validate_origin(candidate)
        seen[normalised] = None  # dict preserves insertion order; dedups
    return tuple(seen.keys())
