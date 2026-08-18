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
import concurrent.futures
import logging
import multiprocessing
import os
import threading
import time
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

# Explicit "spawn", never the platform default. On Linux that default is
# "fork" — cheap (workers inherit already-imported modules via copy-on-
# write), but forking a process that already has other threads running is
# a well-known deadlock hazard: any lock another thread happened to hold at
# fork time is copied into the child still locked, and nothing in the
# child will ever release it (the thread that owned it doesn't exist
# there). hermes serve/hermes dashboard are always multithreaded well
# before this pool's first lazy use (see tui_gateway/server.py's
# threading.Thread-per-turn model), so that hazard is not theoretical here.
# CPython 3.14 reaches the same conclusion generally — it moved off a bare
# "fork" default for exactly this reason. "spawn" starts each worker as a
# genuinely fresh interpreter, sidestepping the hazard entirely, at the
# cost of a slower first-use latency — paid once per process lifetime
# since the pool is a lazy singleton, not per call.
#
# Note this alone was NOT enough to fix the CI hang that motivated the
# hardening below (_hard_shutdown/_SHUTDOWN_GRACE_S) — that hang
# reproduced identically under both "fork" and "spawn". Its actual cause
# was downstream of worker startup, in this pool's shutdown path — see the
# comment above _SHUTDOWN_GRACE_S for the full story. Keeping "spawn"
# regardless, because the fork+threads hazard it avoids is real and
# independently documented, even though it wasn't the hazard behind that
# specific hang.
_MP_CONTEXT = multiprocessing.get_context("spawn")


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
            _pool = ProcessPoolExecutor(max_workers=_POOL_WORKERS, mp_context=_MP_CONTEXT)
        except Exception:
            # Sandboxes / restricted containers sometimes can't fork at all
            # (no /dev/shm, disabled clone()). Falling inline forever is the
            # only sane response — retrying every call would just repeat
            # the same failure on the hot path.
            _log.warning("cpu_offload: could not start process pool, running inline instead", exc_info=True)
            _pool_unavailable = True
            return None
        return _pool


def _discard_pool(known_bad: Executor) -> None:
    """Drop ``_pool`` so the next call lazily rebuilds a fresh one.

    Only clears it if it's still the exact instance that just failed — if
    another thread already swapped in a newer pool (e.g. it lost the same
    race and rebuilt first), this must not clobber that newer, presumably
    healthy one.
    """
    global _pool
    with _pool_lock:
        if _pool is known_bad:
            _pool = None
    _hard_shutdown(known_bad)


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
    except (BrokenProcessPool, concurrent.futures.TimeoutError):
        # Either the pool is confirmed dead (a worker crashed/was killed —
        # BrokenProcessPool), or a call outright timed out. Treat both as
        # pool-poisoning rather than "this one call was just slow": a
        # timed-out call observed in practice was never actually delivered
        # to a worker at all (the workers sat idle in queue.get() the whole
        # time — something in the pool's internal call-queue feeder had
        # already wedged), so every subsequent call would just pay the same
        # full timeout again for nothing. Drop the pool so the next call
        # lazily rebuilds a fresh one, and fall back inline for this call.
        _log.warning("cpu_offload: process pool broken/timed out, rebuilding on next call", exc_info=True)
        _discard_pool(pool)
        return fn(content, *args, **kwargs)
    except Exception:
        # Anything else (e.g. a submit-time pickling error) is scoped to
        # this one call — the pool itself is presumably still healthy.
        _log.warning("cpu_offload: pool call failed, falling back to inline", exc_info=True)
        return fn(content, *args, **kwargs)



# Bound on how long a torn-down pool's worker processes get to exit on
# their own before we force-kill them. This exists because of a real,
# reproduced failure mode, not a hypothetical one: Python's own
# multiprocessing.util atexit hook (registered by the multiprocessing
# module itself the first time any Process is created — independent of
# anything in this module) joins every still-alive non-daemon child with
# NO timeout of its own at interpreter exit. If shutdown() returns while a
# worker is still alive, that hook can then hang the entire interpreter
# forever waiting for it. This reproduced as a genuine CI hang: a large
# real-world payload through code_execution_tool's redact_sensitive_text
# call left worker processes that never received their shutdown sentinel
# — confirmed via py-spy, they sat parked in queue.get() indefinitely, so
# a plain shutdown(wait=True) with no bound would have just moved the same
# hang from the built-in atexit hook into this one. Bounding our own wait
# and force-killing anything left over guarantees this function — and
# therefore the process's exit — can never hang, no matter what has gone
# wrong internally in the pool's own machinery.
_SHUTDOWN_GRACE_S = float(os.environ.get("HERMES_CPU_OFFLOAD_SHUTDOWN_GRACE_S", "5") or "5")


def _hard_shutdown(pool: Executor) -> None:
    # Snapshot the worker Process objects BEFORE calling shutdown(): despite
    # taking a `wait` argument, ProcessPoolExecutor.shutdown() unconditionally
    # sets `self._processes = None` right before it returns, every time,
    # regardless of `wait` — confirmed by reading CPython's
    # concurrent.futures.process source directly. Reading `_processes` after
    # shutdown() (an earlier version of this function did exactly that) gets
    # None, not the pool's workers, and `None.values()` raises AttributeError
    # — silently swallowed by atexit's per-handler exception handling, so
    # this function appeared to run but actually crashed before ever
    # reaching the force-kill loop below, leaving the real hang to happen
    # exactly as if this function didn't exist at all. Grab the reference
    # first and this whole class of bug can't recur.
    #
    # ProcessPoolExecutor-specific: not every Executor (e.g. a test double,
    # or a future ThreadPoolExecutor/InterpreterPoolExecutor swap) has
    # `_processes`, so this is best-effort and silently no-ops otherwise —
    # those alternatives don't carry the non-daemon-child-at-atexit hazard
    # this exists to guard against in the first place.
    processes = list((getattr(pool, "_processes", None) or {}).values())

    pool.shutdown(wait=False, cancel_futures=True)

    deadline = time.monotonic() + _SHUTDOWN_GRACE_S

    for proc in processes:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            proc.join(timeout=remaining)
        if proc.is_alive():
            _log.warning(
                "cpu_offload: worker pid=%s did not exit within %.1fs, killing it",
                proc.pid, _SHUTDOWN_GRACE_S,
            )
            try:
                proc.kill()
                proc.join(timeout=2)
            except Exception:
                _log.warning("cpu_offload: failed to force-kill worker pid=%s", proc.pid, exc_info=True)


def shutdown() -> None:
    """Best-effort pool teardown. Safe to call even if never started.

    Never blocks longer than ``_SHUTDOWN_GRACE_S`` — see :func:`_hard_shutdown`
    for why that bound exists.
    """
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None

    if pool is not None:
        _hard_shutdown(pool)


atexit.register(shutdown)
