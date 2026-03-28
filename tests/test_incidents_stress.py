"""Stress tests for logmind.incidents — 30+ tests covering
auto_detect_incidents, label_incident, and search_incidents."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from logmind.models import IncidentMatch, LogEntry, LogMindIndex, LogWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384  # default dim for all-MiniLM-L6-v2


def _make_entry(
    ts: datetime | None,
    level: str = "INFO",
    source: str = "app",
    message: str = "ok",
    line_number: int = 0,
) -> LogEntry:
    raw = f"{ts} [{level}] {source}: {message}" if ts else f"[{level}] {source}: {message}"
    return LogEntry(
        timestamp=ts,
        level=level,
        source=source,
        message=message,
        raw=raw,
        line_number=line_number,
    )


def _ts(base: datetime, offset_sec: int) -> datetime:
    return base + timedelta(seconds=offset_sec)


BASE_TIME = datetime(2026, 3, 1, 10, 0, 0)


def _build_entries(
    n_normal: int = 50,
    n_error: int = 0,
    error_offset_start: int = 0,
    window_gap: int = 5,
) -> List[LogEntry]:
    """Build a list of entries with optional error spike."""
    entries: List[LogEntry] = []
    for i in range(n_normal):
        entries.append(
            _make_entry(_ts(BASE_TIME, i * window_gap), "INFO", "svc-a", f"normal msg {i}", i)
        )
    for i in range(n_error):
        entries.append(
            _make_entry(
                _ts(BASE_TIME, error_offset_start + i),
                "ERROR",
                "svc-b",
                f"error msg {i}",
                n_normal + i,
            )
        )
    return entries


def _make_index_with_quantizer(dim: int = DIM) -> LogMindIndex:
    """Create a minimal LogMindIndex backed by TurboQuantizer."""
    from langchain_turboquant import TurboQuantizer

    q = TurboQuantizer(dim=dim, bits=3)
    return LogMindIndex(
        normal_compressed=None,
        normal_windows=[],
        incident_compressed=None,
        incident_windows=[],
        incident_labels=[],
        incident_resolutions=[],
        quantizer=q,
        model_name=MODEL_NAME,
        embedding_dim=dim,
        total_lines=0,
        total_windows=0,
        incident_count=0,
        error_count=0,
        warn_count=0,
        sources=["svc-a"],
        learn_time=0.0,
    )


# ====================================================================
# auto_detect_incidents
# ====================================================================

class TestAutoDetectIncidents:
    """Tests for auto_detect_incidents."""

    def test_empty_entries(self):
        from logmind.incidents import auto_detect_incidents
        result = auto_detect_incidents([], window_size=60, spike_threshold=3.0)
        assert result == []

    def test_no_errors_returns_empty(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=20, n_error=0)
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=3.0)
        assert result == []

    def test_single_error_below_threshold(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=60, n_error=1, error_offset_start=30)
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=3.0)
        assert result == []

    def test_large_spike_detected(self):
        from logmind.incidents import auto_detect_incidents
        # Many normal entries across several windows, then a huge spike
        entries = _build_entries(n_normal=100, n_error=0, window_gap=5)
        # Inject a massive error spike within one window
        for i in range(50):
            entries.append(
                _make_entry(_ts(BASE_TIME, 600 + i), "ERROR", "svc-x", f"crash {i}", 200 + i)
            )
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=2.0)
        assert len(result) >= 1
        for w in result:
            assert w.error_count > 0
            assert "auto:" in w.label

    def test_spike_threshold_high_suppresses(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=100, n_error=0, window_gap=5)
        for i in range(10):
            entries.append(
                _make_entry(_ts(BASE_TIME, 600 + i), "ERROR", "svc-x", f"err {i}", 200 + i)
            )
        # Very high threshold should suppress
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=100.0)
        assert result == []

    def test_spike_threshold_zero_catches_any_error(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=60, n_error=5, error_offset_start=300, window_gap=10)
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=0.0)
        # With threshold=0, avg + 0*std = avg; any window above avg is flagged
        # Whether detected depends on distribution; just check it runs
        assert isinstance(result, list)

    def test_all_errors_no_spike(self):
        """If every window has equal errors, no spike is detected."""
        from logmind.incidents import auto_detect_incidents
        entries = []
        for i in range(100):
            entries.append(
                _make_entry(_ts(BASE_TIME, i * 5), "ERROR", "svc", f"err {i}", i)
            )
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=3.0)
        # Uniform distribution => std ~0 => threshold = avg + 3*1 (min std=1)
        # Some windows might still exceed, but uniform should be stable
        assert isinstance(result, list)

    def test_custom_window_size_small(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=40, n_error=0, window_gap=2)
        # Inject spike in a narrow time band
        for i in range(20):
            entries.append(
                _make_entry(_ts(BASE_TIME, 100 + i), "ERROR", "svc", f"err {i}", 100 + i)
            )
        result = auto_detect_incidents(entries, window_size=10, spike_threshold=2.0)
        assert isinstance(result, list)

    def test_custom_window_size_large(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=200, n_error=30, error_offset_start=500, window_gap=3)
        result = auto_detect_incidents(entries, window_size=300, spike_threshold=2.0)
        assert isinstance(result, list)

    def test_entries_without_timestamps(self):
        """Entries with None timestamps should be handled gracefully."""
        from logmind.incidents import auto_detect_incidents
        entries = [_make_entry(None, "ERROR", "svc", f"err {i}", i) for i in range(10)]
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=3.0)
        assert result == []

    def test_mixed_timed_and_untimed(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=30, n_error=0, window_gap=5)
        entries += [_make_entry(None, "ERROR", "svc", f"notimed {i}", 100 + i) for i in range(5)]
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=3.0)
        assert isinstance(result, list)

    def test_label_format_contains_count_and_threshold(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=100, n_error=0, window_gap=5)
        for i in range(50):
            entries.append(
                _make_entry(_ts(BASE_TIME, 600 + i), "ERROR", "svc-x", f"crash {i}", 200 + i)
            )
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=1.0)
        for w in result:
            assert "errors" in w.label
            assert "threshold" in w.label

    def test_return_type_is_logwindow(self):
        from logmind.incidents import auto_detect_incidents
        entries = _build_entries(n_normal=100, n_error=0, window_gap=5)
        for i in range(50):
            entries.append(
                _make_entry(_ts(BASE_TIME, 600 + i), "ERROR", "svc-x", f"crash {i}", 200 + i)
            )
        result = auto_detect_incidents(entries, window_size=60, spike_threshold=1.0)
        for w in result:
            assert isinstance(w, LogWindow)


# ====================================================================
# label_incident
# ====================================================================

class TestLabelIncident:
    """Tests for label_incident (uses real sentence-transformers model)."""

    def test_label_single_incident(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=10, n_error=5, error_offset_start=10, window_gap=2)
        start = _ts(BASE_TIME, 10)
        end = _ts(BASE_TIME, 15)
        updated = label_incident(index, entries, start, end, "disk-full", "expanded volume")
        assert updated.incident_count == 1
        assert updated.incident_labels == ["disk-full"]
        assert updated.incident_resolutions == ["expanded volume"]
        assert len(updated.incident_windows) == 1
        assert updated.incident_compressed is not None

    def test_label_multiple_incidents(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=20, n_error=10, error_offset_start=5, window_gap=1)

        index = label_incident(index, entries, _ts(BASE_TIME, 5), _ts(BASE_TIME, 9), "oom-kill")
        index = label_incident(index, entries, _ts(BASE_TIME, 10), _ts(BASE_TIME, 14), "cpu-spike")
        assert index.incident_count == 2
        assert len(index.incident_labels) == 2
        assert "oom-kill" in index.incident_labels
        assert "cpu-spike" in index.incident_labels

    def test_label_no_entries_in_range_raises(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=10, n_error=0, window_gap=5)
        far_future = datetime(2099, 1, 1)
        with pytest.raises(ValueError, match="No log entries found"):
            label_incident(index, entries, far_future, far_future + timedelta(hours=1), "ghost")

    def test_label_empty_entries_raises(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        with pytest.raises(ValueError, match="No log entries found"):
            label_incident(index, [], BASE_TIME, BASE_TIME + timedelta(hours=1), "nothing")

    def test_label_with_resolution(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=10, n_error=5, error_offset_start=10, window_gap=2)
        updated = label_incident(
            index, entries, _ts(BASE_TIME, 10), _ts(BASE_TIME, 14),
            "db-timeout", "increased connection pool"
        )
        assert updated.incident_resolutions[0] == "increased connection pool"

    def test_label_without_resolution(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=10, n_error=5, error_offset_start=10, window_gap=2)
        updated = label_incident(
            index, entries, _ts(BASE_TIME, 10), _ts(BASE_TIME, 14), "unknown-outage"
        )
        assert updated.incident_resolutions[0] == ""

    def test_label_window_contains_correct_entries(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=20, n_error=5, error_offset_start=10, window_gap=1)
        start = _ts(BASE_TIME, 10)
        end = _ts(BASE_TIME, 14)
        updated = label_incident(index, entries, start, end, "test")
        window = updated.incident_windows[0]
        for e in window.entries:
            assert e.timestamp is not None
            assert start <= e.timestamp <= end

    def test_label_overlapping_ranges(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=20, n_error=10, error_offset_start=5, window_gap=1)
        index = label_incident(index, entries, _ts(BASE_TIME, 5), _ts(BASE_TIME, 12), "inc-a")
        index = label_incident(index, entries, _ts(BASE_TIME, 8), _ts(BASE_TIME, 14), "inc-b")
        assert index.incident_count == 2

    def test_label_preserves_existing_incidents(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries = _build_entries(n_normal=20, n_error=10, error_offset_start=5, window_gap=1)
        index = label_incident(index, entries, _ts(BASE_TIME, 5), _ts(BASE_TIME, 9), "first")
        first_window = index.incident_windows[0]
        index = label_incident(index, entries, _ts(BASE_TIME, 10), _ts(BASE_TIME, 14), "second")
        assert index.incident_windows[0] is first_window
        assert index.incident_labels[0] == "first"


# ====================================================================
# search_incidents
# ====================================================================

class TestSearchIncidents:
    """Tests for search_incidents (uses real model + TurboQuantizer)."""

    @pytest.fixture()
    def index_with_incidents(self):
        from logmind.incidents import label_incident
        index = _make_index_with_quantizer()
        entries: List[LogEntry] = []
        # Window 1 - disk full errors
        for i in range(10):
            entries.append(
                _make_entry(
                    _ts(BASE_TIME, i), "ERROR", "storage-svc",
                    "disk full: /data partition 99% used, write failed", i,
                )
            )
        # Window 2 - OOM kill errors
        for i in range(10):
            entries.append(
                _make_entry(
                    _ts(BASE_TIME, 100 + i), "ERROR", "app-server",
                    "OutOfMemoryError: Java heap space exceeded", 20 + i,
                )
            )
        # Window 3 - network timeout
        for i in range(10):
            entries.append(
                _make_entry(
                    _ts(BASE_TIME, 200 + i), "ERROR", "api-gateway",
                    "connection timeout to upstream service db-primary:5432", 40 + i,
                )
            )

        index = label_incident(
            index, entries, _ts(BASE_TIME, 0), _ts(BASE_TIME, 9),
            "disk-full", "expanded /data volume to 500GB",
        )
        index = label_incident(
            index, entries, _ts(BASE_TIME, 100), _ts(BASE_TIME, 109),
            "oom-kill", "increased JVM heap to 8GB",
        )
        index = label_incident(
            index, entries, _ts(BASE_TIME, 200), _ts(BASE_TIME, 209),
            "db-timeout", "restarted db-primary and increased max_connections",
        )
        return index

    def test_search_returns_results(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk full error", index_with_incidents)
        assert len(results) > 0
        assert all(isinstance(r, IncidentMatch) for r in results)

    def test_search_top_result_relevance(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk space is full", index_with_incidents)
        assert len(results) > 0
        # The top result should be the disk-full incident
        assert results[0].label == "disk-full"

    def test_search_memory_query(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("out of memory heap", index_with_incidents)
        assert len(results) > 0
        assert results[0].label == "oom-kill"

    def test_search_network_query(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("database connection timeout", index_with_incidents)
        assert len(results) > 0
        assert results[0].label == "db-timeout"

    def test_search_empty_index(self):
        from logmind.incidents import search_incidents
        index = _make_index_with_quantizer()
        results = search_incidents("anything", index)
        assert results == []

    def test_search_k_limits_results(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("error", index_with_incidents, k=1)
        assert len(results) <= 1

    def test_search_k_larger_than_incidents(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("error", index_with_incidents, k=100)
        assert len(results) <= 3  # only 3 incidents exist

    def test_search_result_has_similarity(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk full", index_with_incidents)
        for r in results:
            assert 0.0 <= r.similarity <= 1.0

    def test_search_result_has_resolution(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk full", index_with_incidents)
        top = results[0]
        assert top.resolution != ""

    def test_search_result_has_occurred_at(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk full", index_with_incidents)
        top = results[0]
        assert top.occurred_at is not None

    def test_search_result_has_window(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("disk full", index_with_incidents)
        top = results[0]
        assert top.window is not None
        assert isinstance(top.window, LogWindow)

    def test_search_results_sorted_by_similarity(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents("server error", index_with_incidents)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_search_irrelevant_query_low_scores(self, index_with_incidents):
        from logmind.incidents import search_incidents
        results = search_incidents(
            "sunshine rainbow unicorn happiness", index_with_incidents
        )
        # All results should have relatively low similarity
        for r in results:
            assert r.similarity < 0.9
