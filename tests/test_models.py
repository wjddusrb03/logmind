"""Tests for data models."""

import pytest
from datetime import datetime
from logmind.models import LogEntry, LogWindow, AnomalyAlert, IncidentMatch


class TestLogEntry:
    def test_is_error(self):
        e = LogEntry(None, "ERROR", "app", "fail", "raw")
        assert e.is_error is True
        e2 = LogEntry(None, "FATAL", "app", "crash", "raw")
        assert e2.is_error is True

    def test_is_not_error(self):
        e = LogEntry(None, "INFO", "app", "ok", "raw")
        assert e.is_error is False

    def test_is_warn(self):
        e = LogEntry(None, "WARN", "app", "slow", "raw")
        assert e.is_warn is True
        e2 = LogEntry(None, "WARNING", "app", "slow", "raw")
        assert e2.is_warn is True

    def test_defaults(self):
        e = LogEntry(None, "INFO", "app", "msg", "raw")
        assert e.metadata == {}
        assert e.line_number == 0


class TestLogWindow:
    def _make_entries(self, levels):
        return [
            LogEntry(datetime(2024, 1, 1, 10, 0, i), level, f"src{i}", f"msg{i}", f"raw{i}")
            for i, level in enumerate(levels)
        ]

    def test_error_count(self):
        entries = self._make_entries(["INFO", "ERROR", "ERROR", "WARN"])
        w = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
        assert w.error_count == 2

    def test_warn_count(self):
        entries = self._make_entries(["INFO", "WARN", "WARNING", "ERROR"])
        w = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
        assert w.warn_count == 2

    def test_total_count(self):
        entries = self._make_entries(["INFO", "ERROR"])
        w = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
        assert w.total_count == 2

    def test_source_distribution(self):
        entries = self._make_entries(["INFO", "ERROR", "WARN"])
        w = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
        dist = w.source_distribution
        assert len(dist) == 3

    def test_is_incident(self):
        w = LogWindow([], None, None, label="outage")
        assert w.is_incident is True
        w2 = LogWindow([], None, None)
        assert w2.is_incident is False

    def test_to_embedding_text(self):
        entries = self._make_entries(["ERROR", "WARN", "INFO"])
        w = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
        text = w.to_embedding_text()
        assert "errors=1" in text
        assert "warns=1" in text
        assert "[ERROR]" in text


class TestAnomalyAlert:
    def test_top_incident(self):
        inc = IncidentMatch(0.9, "DB down", "restart", datetime(2024,1,1))
        alert = AnomalyAlert(
            LogWindow([], None, None), 0.8, "similar_incident",
            [inc], "CRITICAL", "summary"
        )
        assert alert.top_incident == inc

    def test_no_incidents(self):
        alert = AnomalyAlert(
            LogWindow([], None, None), 0.5, "new_pattern",
            [], "WARNING", "summary"
        )
        assert alert.top_incident is None
