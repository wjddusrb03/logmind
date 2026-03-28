"""Rigorous exhaustive tests for logmind.models — 40+ tests covering
every level, edge case, and property of LogEntry, LogWindow, AnomalyAlert,
IncidentMatch, and LogMindIndex."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime, timedelta
from typing import List

import pytest

from logmind.models import (
    AnomalyAlert,
    IncidentMatch,
    LogEntry,
    LogMindIndex,
    LogWindow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 3, 1, 12, 0, 0)


def _entry(
    level: str = "INFO",
    source: str = "app",
    message: str = "ok",
    ts: datetime | None = BASE_TIME,
    metadata: dict | None = None,
    line_number: int = 0,
) -> LogEntry:
    raw = f"[{level}] {source}: {message}"
    return LogEntry(
        timestamp=ts,
        level=level,
        source=source,
        message=message,
        raw=raw,
        metadata=metadata or {},
        line_number=line_number,
    )


def _window(
    entries: List[LogEntry] | None = None,
    label: str = "",
    resolution: str = "",
) -> LogWindow:
    if entries is None:
        entries = []
    return LogWindow(
        entries=entries,
        start_time=BASE_TIME,
        end_time=BASE_TIME + timedelta(seconds=60),
        label=label,
        resolution=resolution,
    )


# ====================================================================
# LogEntry — is_error exhaustive per level
# ====================================================================

class TestLogEntryIsError:
    def test_error_level(self):
        assert _entry(level="ERROR").is_error is True

    def test_fatal_level(self):
        assert _entry(level="FATAL").is_error is True

    def test_critical_level(self):
        assert _entry(level="CRITICAL").is_error is True

    def test_warn_not_error(self):
        assert _entry(level="WARN").is_error is False

    def test_warning_not_error(self):
        assert _entry(level="WARNING").is_error is False

    def test_info_not_error(self):
        assert _entry(level="INFO").is_error is False

    def test_debug_not_error(self):
        assert _entry(level="DEBUG").is_error is False

    def test_trace_not_error(self):
        assert _entry(level="TRACE").is_error is False

    def test_unknown_not_error(self):
        assert _entry(level="UNKNOWN").is_error is False

    def test_empty_string_not_error(self):
        assert _entry(level="").is_error is False


# ====================================================================
# LogEntry — is_warn exhaustive per level
# ====================================================================

class TestLogEntryIsWarn:
    def test_warn_level(self):
        assert _entry(level="WARN").is_warn is True

    def test_warning_level(self):
        assert _entry(level="WARNING").is_warn is True

    def test_error_not_warn(self):
        assert _entry(level="ERROR").is_warn is False

    def test_fatal_not_warn(self):
        assert _entry(level="FATAL").is_warn is False

    def test_critical_not_warn(self):
        assert _entry(level="CRITICAL").is_warn is False

    def test_info_not_warn(self):
        assert _entry(level="INFO").is_warn is False

    def test_debug_not_warn(self):
        assert _entry(level="DEBUG").is_warn is False

    def test_trace_not_warn(self):
        assert _entry(level="TRACE").is_warn is False

    def test_unknown_not_warn(self):
        assert _entry(level="UNKNOWN").is_warn is False

    def test_empty_string_not_warn(self):
        assert _entry(level="").is_warn is False


# ====================================================================
# LogEntry — metadata & line_number edge cases
# ====================================================================

class TestLogEntryFields:
    def test_metadata_default_empty(self):
        e = LogEntry(None, "INFO", "app", "msg", "raw")
        assert e.metadata == {}

    def test_metadata_with_various_types(self):
        meta = {"key": "value", "number": "42", "empty": ""}
        e = _entry(metadata=meta)
        assert e.metadata == meta

    def test_metadata_unicode_values(self):
        meta = {"lang": "한국어", "emoji": "fire"}
        e = _entry(metadata=meta)
        assert e.metadata["lang"] == "한국어"

    def test_line_number_default_zero(self):
        e = LogEntry(None, "INFO", "app", "msg", "raw")
        assert e.line_number == 0

    def test_line_number_zero(self):
        e = _entry(line_number=0)
        assert e.line_number == 0

    def test_line_number_negative(self):
        e = _entry(line_number=-1)
        assert e.line_number == -1

    def test_line_number_very_large(self):
        e = _entry(line_number=10_000_000)
        assert e.line_number == 10_000_000

    def test_timestamp_none(self):
        e = _entry(ts=None)
        assert e.timestamp is None

    def test_timestamp_set(self):
        ts = datetime(2026, 1, 15, 8, 0, 0)
        e = _entry(ts=ts)
        assert e.timestamp == ts


# ====================================================================
# LogWindow — error_count / warn_count / total_count
# ====================================================================

class TestLogWindowCounts:
    def test_counts_empty(self):
        w = _window(entries=[])
        assert w.error_count == 0
        assert w.warn_count == 0
        assert w.total_count == 0

    def test_counts_single_error(self):
        w = _window(entries=[_entry(level="ERROR")])
        assert w.error_count == 1
        assert w.warn_count == 0
        assert w.total_count == 1

    def test_counts_100_entries_mixed(self):
        entries = (
            [_entry(level="ERROR")] * 30
            + [_entry(level="WARN")] * 20
            + [_entry(level="INFO")] * 50
        )
        w = _window(entries=entries)
        assert w.error_count == 30
        assert w.warn_count == 20
        assert w.total_count == 100

    def test_counts_1000_entries(self):
        entries = [_entry(level="FATAL")] * 1000
        w = _window(entries=entries)
        assert w.error_count == 1000
        assert w.warn_count == 0
        assert w.total_count == 1000

    def test_all_warn_variants_counted(self):
        entries = [_entry(level="WARN"), _entry(level="WARNING")]
        w = _window(entries=entries)
        assert w.warn_count == 2

    def test_all_error_variants_counted(self):
        entries = [
            _entry(level="ERROR"),
            _entry(level="FATAL"),
            _entry(level="CRITICAL"),
        ]
        w = _window(entries=entries)
        assert w.error_count == 3


# ====================================================================
# LogWindow — source_distribution
# ====================================================================

class TestLogWindowSourceDistribution:
    def test_empty_entries(self):
        w = _window(entries=[])
        assert w.source_distribution == {}

    def test_single_source(self):
        entries = [_entry(source="api")] * 5
        w = _window(entries=entries)
        assert w.source_distribution == {"api": 5}

    def test_duplicate_sources(self):
        entries = [
            _entry(source="api"),
            _entry(source="api"),
            _entry(source="db"),
            _entry(source="db"),
            _entry(source="db"),
        ]
        w = _window(entries=entries)
        dist = w.source_distribution
        assert dist["api"] == 2
        assert dist["db"] == 3

    def test_many_distinct_sources(self):
        entries = [_entry(source=f"svc-{i}") for i in range(20)]
        w = _window(entries=entries)
        assert len(w.source_distribution) == 20

    def test_empty_source_string(self):
        entries = [_entry(source="")]
        w = _window(entries=entries)
        assert w.source_distribution == {"": 1}


# ====================================================================
# LogWindow — to_embedding_text
# ====================================================================

class TestLogWindowToEmbeddingText:
    def test_empty_window(self):
        w = _window(entries=[])
        text = w.to_embedding_text()
        assert "errors=0" in text
        assert "warns=0" in text
        assert "total=0" in text

    def test_all_errors(self):
        entries = [_entry(level="ERROR", message=f"err-{i}") for i in range(5)]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "errors=5" in text
        assert "[ERROR]" in text

    def test_all_info_no_important_messages(self):
        entries = [_entry(level="INFO", message=f"info-{i}") for i in range(3)]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "errors=0" in text
        assert "[ERROR]" not in text
        assert "[INFO]" not in text  # INFO not included in important

    def test_mixed_levels(self):
        entries = [
            _entry(level="ERROR", source="db", message="conn lost"),
            _entry(level="WARN", source="api", message="slow query"),
            _entry(level="INFO", source="web", message="request ok"),
        ]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "errors=1" in text
        assert "warns=1" in text
        assert "total=3" in text
        assert "[ERROR] db: conn lost" in text
        assert "[WARN] api: slow query" in text

    def test_sources_top5(self):
        entries = [_entry(source=f"svc-{i}", level="ERROR") for i in range(8)]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "Sources:" in text

    def test_dedup_similar_messages_capped_at_50(self):
        # 60 entries with identical first-100-chars messages -> deduped to 1
        entries = [_entry(level="ERROR", message="same error") for _ in range(60)]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        error_lines = [ln for ln in text.split("\n") if ln.startswith("[ERROR]")]
        assert len(error_lines) == 1  # deduplication

    def test_dedup_distinct_messages(self):
        entries = [_entry(level="ERROR", message=f"unique-{i}") for i in range(10)]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        error_lines = [ln for ln in text.split("\n") if ln.startswith("[ERROR]")]
        assert len(error_lines) == 10

    def test_dedup_key_truncates_at_100_chars(self):
        base = "A" * 100
        entries = [
            _entry(level="ERROR", message=base + "-suffix1"),
            _entry(level="ERROR", message=base + "-suffix2"),
        ]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        error_lines = [ln for ln in text.split("\n") if ln.startswith("[ERROR]")]
        # First 100 chars identical -> deduped to 1
        assert len(error_lines) == 1

    def test_includes_fatal_and_critical(self):
        entries = [
            _entry(level="FATAL", message="fatal crash"),
            _entry(level="CRITICAL", message="critical issue"),
        ]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "[FATAL]" in text
        assert "[CRITICAL]" in text

    def test_includes_warning_level(self):
        entries = [_entry(level="WARNING", message="deprecated")]
        w = _window(entries=entries)
        text = w.to_embedding_text()
        assert "[WARNING]" in text


# ====================================================================
# LogWindow — is_incident
# ====================================================================

class TestLogWindowIsIncident:
    def test_empty_label(self):
        w = _window(label="")
        assert w.is_incident is False

    def test_whitespace_label(self):
        w = _window(label="   ")
        assert w.is_incident is True  # bool("   ") is True

    def test_real_label(self):
        w = _window(label="database-outage")
        assert w.is_incident is True

    def test_default_label(self):
        w = LogWindow(entries=[], start_time=None, end_time=None)
        assert w.is_incident is False


# ====================================================================
# LogWindow — defaults
# ====================================================================

class TestLogWindowDefaults:
    def test_window_size_default(self):
        w = LogWindow(entries=[], start_time=None, end_time=None)
        assert w.window_size == 60

    def test_label_default(self):
        w = LogWindow(entries=[], start_time=None, end_time=None)
        assert w.label == ""

    def test_resolution_default(self):
        w = LogWindow(entries=[], start_time=None, end_time=None)
        assert w.resolution == ""


# ====================================================================
# AnomalyAlert — top_incident
# ====================================================================

class TestAnomalyAlertTopIncident:
    def _alert_with_incidents(self, incs: list) -> AnomalyAlert:
        return AnomalyAlert(
            current_window=_window(),
            anomaly_score=0.8,
            anomaly_type="error_spike",
            similar_incidents=incs,
            severity="WARNING",
            summary="test",
        )

    def test_no_incidents(self):
        alert = self._alert_with_incidents([])
        assert alert.top_incident is None

    def test_single_incident(self):
        inc = IncidentMatch(0.9, "label", "res", None)
        alert = self._alert_with_incidents([inc])
        assert alert.top_incident is inc

    def test_multiple_incidents_returns_first(self):
        inc1 = IncidentMatch(0.95, "first", "fix1", None)
        inc2 = IncidentMatch(0.80, "second", "fix2", None)
        inc3 = IncidentMatch(0.50, "third", "fix3", None)
        alert = self._alert_with_incidents([inc1, inc2, inc3])
        assert alert.top_incident is inc1


# ====================================================================
# IncidentMatch — field combinations
# ====================================================================

class TestIncidentMatchFields:
    def test_all_populated(self):
        ts = datetime(2026, 1, 1)
        w = _window()
        inc = IncidentMatch(
            similarity=0.92,
            label="outage",
            resolution="restart",
            occurred_at=ts,
            window=w,
        )
        assert inc.similarity == 0.92
        assert inc.label == "outage"
        assert inc.resolution == "restart"
        assert inc.occurred_at == ts
        assert inc.window is w

    def test_all_none_or_empty(self):
        inc = IncidentMatch(
            similarity=0.0,
            label="",
            resolution="",
            occurred_at=None,
            window=None,
        )
        assert inc.similarity == 0.0
        assert inc.label == ""
        assert inc.resolution == ""
        assert inc.occurred_at is None
        assert inc.window is None

    def test_window_default_none(self):
        inc = IncidentMatch(0.5, "lbl", "res", None)
        assert inc.window is None


# ====================================================================
# LogMindIndex — defaults
# ====================================================================

class TestLogMindIndexDefaults:
    def test_avg_errors_default(self):
        from langchain_turboquant import TurboQuantizer
        q = TurboQuantizer(dim=32, bits=3)
        idx = LogMindIndex(
            normal_compressed=None,
            normal_windows=[],
            incident_compressed=None,
            incident_windows=[],
            incident_labels=[],
            incident_resolutions=[],
            quantizer=q,
            model_name="test",
            embedding_dim=32,
            total_lines=0,
            total_windows=0,
            incident_count=0,
            error_count=0,
            warn_count=0,
            sources=[],
            learn_time=0.0,
        )
        assert idx.avg_errors_per_window == 0.0
        assert idx.std_errors_per_window == 0.0
        assert idx.avg_warns_per_window == 0.0

    def test_custom_avg_std(self):
        from langchain_turboquant import TurboQuantizer
        q = TurboQuantizer(dim=32, bits=3)
        idx = LogMindIndex(
            normal_compressed=None,
            normal_windows=[],
            incident_compressed=None,
            incident_windows=[],
            incident_labels=[],
            incident_resolutions=[],
            quantizer=q,
            model_name="test",
            embedding_dim=32,
            total_lines=100,
            total_windows=10,
            incident_count=2,
            error_count=50,
            warn_count=20,
            sources=["a"],
            learn_time=1.0,
            avg_errors_per_window=5.5,
            std_errors_per_window=2.3,
            avg_warns_per_window=1.7,
        )
        assert idx.avg_errors_per_window == 5.5
        assert idx.std_errors_per_window == 2.3
        assert idx.avg_warns_per_window == 1.7


# ====================================================================
# Dataclass field defaults — verify all defaults work correctly
# ====================================================================

class TestDataclassDefaults:
    def test_log_entry_has_expected_fields(self):
        names = {f.name for f in dc_fields(LogEntry)}
        assert "timestamp" in names
        assert "level" in names
        assert "source" in names
        assert "message" in names
        assert "raw" in names
        assert "metadata" in names
        assert "line_number" in names

    def test_log_window_has_expected_fields(self):
        names = {f.name for f in dc_fields(LogWindow)}
        assert "entries" in names
        assert "start_time" in names
        assert "end_time" in names
        assert "window_size" in names
        assert "label" in names
        assert "resolution" in names

    def test_incident_match_window_default(self):
        inc = IncidentMatch(0.5, "l", "r", None)
        assert inc.window is None

    def test_logmind_index_default_floats(self):
        """All three float defaults should be 0.0."""
        from dataclasses import MISSING

        # Just verify the dataclass can be constructed with defaults
        from langchain_turboquant import TurboQuantizer
        q = TurboQuantizer(dim=32, bits=3)
        idx = LogMindIndex(
            normal_compressed=None, normal_windows=[], incident_compressed=None,
            incident_windows=[], incident_labels=[], incident_resolutions=[],
            quantizer=q, model_name="m", embedding_dim=32, total_lines=0,
            total_windows=0, incident_count=0, error_count=0, warn_count=0,
            sources=[], learn_time=0.0,
        )
        assert idx.avg_errors_per_window == 0.0
        assert idx.std_errors_per_window == 0.0
        assert idx.avg_warns_per_window == 0.0

    def test_metadata_not_shared_between_instances(self):
        """Each LogEntry should get its own dict, not a shared one."""
        e1 = LogEntry(None, "INFO", "a", "m", "r")
        e2 = LogEntry(None, "INFO", "a", "m", "r")
        e1.metadata["key"] = "val"
        assert "key" not in e2.metadata
