"""Tests for tools/cpu_offload.py — the ProcessPoolExecutor offload helper
backing tools.threat_patterns.scan_for_threats and
agent.redact.redact_sensitive_text.

Priorities, in order: (1) small content never pays pool overhead, (2) large
content actually gets offloaded and still returns the correct result, (3)
every pool failure mode falls back to running inline rather than skipping
the work or raising — the fail-closed property this module exists for.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock

import pytest

from tools import cpu_offload


# A plain module-level function so it's picklable across the real process
# pool in the "actually offloads" test below.
def _upper(text: str, suffix: str = "") -> str:
    return text.upper() + suffix


@pytest.fixture(autouse=True)
def _reset_pool_state():
    """Every test starts from a clean slate and leaves one behind."""
    cpu_offload._pool = None
    cpu_offload._pool_unavailable = False
    yield
    cpu_offload.shutdown()
    cpu_offload._pool_unavailable = False


class TestThresholdGating:
    def test_below_threshold_runs_inline_without_touching_pool(self, monkeypatch):
        get_pool = MagicMock(side_effect=AssertionError("pool should not be consulted for small content"))
        monkeypatch.setattr(cpu_offload, "_get_pool", get_pool)

        result = cpu_offload.offload(_upper, "short", threshold=100)

        assert result == "SHORT"
        get_pool.assert_not_called()

    def test_at_or_above_threshold_consults_the_pool(self, monkeypatch):
        sentinel_pool = MagicMock()
        future = MagicMock()
        future.result.return_value = "OFFLOADED"
        sentinel_pool.submit.return_value = future
        monkeypatch.setattr(cpu_offload, "_get_pool", lambda: sentinel_pool)

        result = cpu_offload.offload(_upper, "x" * 10, threshold=10)

        assert result == "OFFLOADED"
        sentinel_pool.submit.assert_called_once()


class TestRealPool:
    def test_actually_offloads_and_returns_correct_result(self):
        payload = "hello world " * 50
        result = cpu_offload.offload(_upper, payload, "!", threshold=10)

        assert result == payload.upper() + "!"
        # A real ProcessPoolExecutor should have been created for this.
        assert cpu_offload._pool is not None


class TestFailClosed:
    """Every failure mode must still return the correct (inline) result —
    never an empty/unmodified passthrough, and never an exception."""

    def test_pool_creation_failure_falls_back_inline(self, monkeypatch):
        monkeypatch.setattr(
            cpu_offload,
            "ProcessPoolExecutor",
            MagicMock(side_effect=OSError("no fork available in this sandbox")),
        )

        result = cpu_offload.offload(_upper, "x" * 20, threshold=10)

        assert result == ("x" * 20).upper()
        assert cpu_offload._pool_unavailable is True

        # And it stays inline on a second call rather than retrying the
        # failing constructor on every hot-path invocation.
        ctor = MagicMock(side_effect=AssertionError("must not retry a known-bad pool"))
        monkeypatch.setattr(cpu_offload, "ProcessPoolExecutor", ctor)
        result2 = cpu_offload.offload(_upper, "y" * 20, threshold=10)
        assert result2 == ("y" * 20).upper()
        ctor.assert_not_called()

    def test_broken_process_pool_falls_back_and_self_heals(self, monkeypatch):
        broken_pool = MagicMock()
        broken_pool.submit.side_effect = concurrent.futures.process.BrokenProcessPool("worker died")
        monkeypatch.setattr(cpu_offload, "_get_pool", lambda: broken_pool)
        cpu_offload._pool = broken_pool

        result = cpu_offload.offload(_upper, "x" * 20, threshold=10)

        assert result == ("x" * 20).upper()
        # The dead pool must be dropped so the next call can rebuild one,
        # not wedge forever on a corpse.
        assert cpu_offload._pool is None

    def test_submit_time_exception_falls_back_without_killing_the_pool(self, monkeypatch):
        flaky_pool = MagicMock()
        flaky_pool.submit.side_effect = TypeError("cannot pickle a weird arg")
        monkeypatch.setattr(cpu_offload, "_get_pool", lambda: flaky_pool)
        cpu_offload._pool = flaky_pool

        result = cpu_offload.offload(_upper, "x" * 20, threshold=10)

        assert result == ("x" * 20).upper()
        # This failure was scoped to the call, not the pool — a healthy
        # pool object should not have been discarded.
        assert cpu_offload._pool is flaky_pool

    def test_result_timeout_falls_back_inline(self, monkeypatch):
        hung_pool = MagicMock()
        future = MagicMock()
        future.result.side_effect = concurrent.futures.TimeoutError()
        hung_pool.submit.return_value = future
        monkeypatch.setattr(cpu_offload, "_get_pool", lambda: hung_pool)

        result = cpu_offload.offload(_upper, "x" * 20, threshold=10)

        assert result == ("x" * 20).upper()
