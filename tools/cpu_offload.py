"""Shared CPU-bound-offload helper for security-scanning hot paths.

``hermes serve``/``hermes dashboard`` run every session's agent turn as an
in-process Python thread sharing one asyncio event loop (see
``tui_gateway/server.py``). CPython's GIL means any thread doing real
CPU-bound work — as opposed to I/O wait, which releases the GIL — competes
with the event loop for execution time; under enough concurrent load the
loop can go unserviced for seconds at a stretch (``WSTransport.write()``'s
"loop stalled" warning in ``tui_gateway/ws.py`` measures exactly this).

Profiling a live dashboard under concurrent load identified
``tools.threat_patterns.scan_for_threats`` and
``agent.redact.redact_sensitive_text`` as the two largest GIL-holding
contributors — both re-scan potentially large tool-output/context text with
a battery of regexes on essentially every message. This module offloads
that specific work to a small persistent ``ProcessPoolExecutor``, which
gets real OS-level parallelism (separate processes, no shared GIL) instead
of contending with the event loop thread.

Written against the plain ``concurrent.futures.Executor`` interface
(``.submit()``) rather than anything ``ProcessPoolExecutor``-specific, so
swapping the pool implementation later is a one-line change, not a
call-site rewrite:

- ``concurrent.futures.ThreadPoolExecutor``, once free-threaded Python is a
  viable target for hermes-agent's dependency set — it currently isn't
  (see the ``requires-python`` comment in ``pyproject.toml``: several
  transitive C-extension deps don't yet build against the free-threaded
  ABI).
- ``concurrent.futures.InterpreterPoolExecutor`` (subinterpreters, PEP 684's
  per-interpreter GIL — true parallelism *without* free-threading), once
  the project's minimum Python floor reaches 3.14. It doesn't exist before
  3.14 at all (verified: absent in 3.11/3.12/3.13), so it isn't an option
  yet either.

``ProcessPoolExecutor`` is the one option available on every Python this
project already supports (>=3.11).
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")

# Below this length, round-tripping content through a worker process
# (pickle out, pickle the result back) costs more than just running the
# regex work inline — offloading only pays off once the content itself is
# expensive to scan. 8KB comfortably clears "a short chat message" while
# catching "a file/tool-output landed in context", which is the case
# profiling showed actually costing multiple milliseconds of GIL-held time.
# Both knobs are env-configurable for operators who want to tune the
# tradeoff for their own workload without a code change.
OFFLOAD_THRESHOLD_CHARS = int(os.environ.get("HERMES_CPU_OFFLOAD_THRESHOLD_CHARS", "8192") or "8192")

# Small and lazy by design: most `hermes` invocations are short-lived CLI
# commands that never touch a content-heavy scan/redact call, so nothing
# should be spun up until first genuine use.
_POOL_WORKERS = max(1, int(os.environ.get("HERMES_CPU_OFFLOAD_POOL_WORKERS", "2") or "2"))

# Generous relative to the regex work itself (profiling showed ~20-30ms for
# a 120KB payload) but bounded, so a wedged worker can't hang the caller's
# thread forever — the caller falls back to running inline instead.
_OFFLOAD_TIMEOUT_S = float(os.environ.get("HERMES_CPU_OFFLOAD_TIMEOUT_S", "20") or "20")

_pool: Executor | None = None
_pool_lock = threading.Lock()
_pool_unavailable = False


def _get_pool() -> Executor | None:
    """Return the shared process pool, creating it lazily on first use.

    Returns ``None`` (never raises) if the pool could not be created —
    callers must treat that as "run inline", not as a reason to skip the
    work; see :func:`offload`.
    """
    global _pool, _pool_unavailable

    if _pool_unavailable:
        return None
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None or _pool_unavailable:
            return _pool
        try:
            _pool = ProcessPoolExecutor(max_workers=_POOL_WORKERS)
        except Exception:
            # Sandboxes / restricted containers sometimes can't fork at all
            # (no /dev/shm, disabled clone()). Falling inline forever is the
            # only sane response — retrying every call would just repeat
            # the same failure on the hot path.
            _log.warning("cpu_offload: could not start process pool, running inline instead", exc_info=True)
            _pool_unavailable = True
            return None
        return _pool


def offload(fn: Callable[..., T], content: str, /, *args, threshold: int = OFFLOAD_THRESHOLD_CHARS, **kwargs) -> T:
    """Run ``fn(content, *args, **kwargs)``, offloaded to a process pool when
    ``content`` is large enough to make that worthwhile.

    Fail-closed: any problem getting a result from the pool (no pool
    available, a crashed/broken worker, a hung call, an unpicklable
    argument) falls back to calling ``fn`` inline in the caller's own
    thread/process rather than skipping the work or raising — this backs
    security-sensitive scanners (threat detection, secret redaction), where
    silently skipping the scan is a worse outcome than running it slow.

    ``fn`` must be a module-level callable (picklable by reference) and its
    arguments must be picklable; both hold for the plain-string,
    stdlib-only functions this module was built for.
    """
    if len(content) < threshold:
        return fn(content, *args, **kwargs)

    pool = _get_pool()
    if pool is None:
        return fn(content, *args, **kwargs)

    try:
        future = pool.submit(fn, content, *args, **kwargs)
        return future.result(timeout=_OFFLOAD_TIMEOUT_S)
    except BrokenProcessPool:
        # The pool itself is dead (a worker crashed/was killed) — every
        # future submit would fail the same way. Drop it so the next call
        # lazily rebuilds a fresh one instead of wedging on a corpse pool
        # forever, and fall back inline for this call.
        _log.warning("cpu_offload: process pool broken, rebuilding on next call", exc_info=True)
        global _pool
        with _pool_lock:
            _pool = None
        return fn(content, *args, **kwargs)
    except Exception:
        # Anything else (submit-time pickling error, a timeout, ...) is
        # scoped to this one call — the pool itself is still healthy.
        _log.warning("cpu_offload: pool call failed, falling back to inline", exc_info=True)
        return fn(content, *args, **kwargs)


def shutdown() -> None:
    """Best-effort pool teardown. Safe to call even if never started."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


atexit.register(shutdown)
