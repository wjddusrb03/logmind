"""Rigorous exhaustive tests for logmind.display — 40+ tests covering
exact format strings, JSON consistency, edge cases, and every display function."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List

import pytest

from langchain_turboquant import TurboQuantizer
from logmind.display import (
    _severity_color,
    _severity_icon,
    display_alert,
    display_scan_report,
    display_search_results,
    display_stats,
)
from logmind.models import AnomalyAlert, IncidentMatch, LogEntry, LogMindIndex, LogWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32
BITS = 3
BASE_TIME = datetime(2026, 3, 1, 12, 0, 0)


def _entry(
    level: str = "INFO",
    source: str = "app",
    message: str = "ok",
    ts: datetime | None = BASE_TIME,
) -> LogEntry:
    return LogEntry(
        timestamp=ts, level=level, source=source,
        message=message, raw=f"[{level}] {source}: {message}",
    )


def _window(
    entries: List[LogEntry] | None = None,
    start: datetime | None = BASE_TIME,
    label: str = "",
    resolution: str = "",
) -> LogWindow:
    if entries is None:
        entries = []
    end = start + timedelta(seconds=60) if start else None
    return LogWindow(entries=entries, start_time=start, end_time=end,
                     label=label, resolution=resolution)


def _incident(
    similarity: float = 0.85,
    label: str = "disk-full",
    resolution: str = "expanded volume",
    occurred_at: datetime | None = BASE_TIME - timedelta(days=7),
) -> IncidentMatch:
    return IncidentMatch(similarity=similarity, label=label,
                         resolution=resolution, occurred_at=occurred_at)


def _alert(
    severity: str = "WARNING",
    score: float = 0.75,
    anomaly_type: str = "error_spike",
    summary: str = "Error rate spiked",
    incidents: List[IncidentMatch] | None = None,
    window: LogWindow | None = None,
) -> AnomalyAlert:
    if window is None:
        window = _window(entries=[_entry(level="ERROR")])
    return AnomalyAlert(
        current_window=window, anomaly_score=score,
        anomaly_type=anomaly_type, similar_incidents=incidents or [],
        severity=severity, summary=summary,
    )


def _index(**overrides) -> LogMindIndex:
    q = TurboQuantizer(dim=DIM, bits=BITS)
    defaults = dict(
        normal_compressed=None, normal_windows=[],
        incident_compressed=None, incident_windows=[],
        incident_labels=[], incident_resolutions=[],
        quantizer=q, model_name="test-model", embedding_dim=DIM,
        total_lines=5000, total_windows=100, incident_count=0,
        error_count=200, warn_count=80, sources=["api", "db"],
        learn_time=4.2,
    )
    defaults.update(overrides)
    return LogMindIndex(**defaults)


# ====================================================================
# display_alert — exact severity icon verification
# ====================================================================

class TestDisplayAlertSeverityIcons:
    def test_critical_shows_triple_bang(self):
        out = display_alert(_alert(severity="CRITICAL"))
        assert "[!!!]" in out
        assert "CRITICAL" in out

    def test_warning_shows_double_bang(self):
        out = display_alert(_alert(severity="WARNING"))
        assert "[!!]" in out
        assert "WARNING" in out

    def test_info_shows_i(self):
        out = display_alert(_alert(severity="INFO"))
        assert "[i]" in out
        assert "INFO" in out

    def test_unknown_severity_shows_question(self):
        out = display_alert(_alert(severity="UNKNOWN"))
        assert "[?]" in out


# ====================================================================
# display_alert — similar incidents count variations
# ====================================================================

class TestDisplayAlertIncidentCounts:
    def test_zero_incidents(self):
        out = display_alert(_alert(incidents=[]))
        assert "Similar Past Incidents:" not in out

    def test_one_incident(self):
        out = display_alert(_alert(incidents=[_incident(label="inc-1")]))
        assert "Similar Past Incidents:" in out
        assert "inc-1" in out

    def test_two_incidents(self):
        incs = [_incident(label="inc-1"), _incident(label="inc-2")]
        out = display_alert(_alert(incidents=incs))
        assert "inc-1" in out
        assert "inc-2" in out

    def test_five_incidents(self):
        incs = [_incident(label=f"inc-{i}") for i in range(5)]
        out = display_alert(_alert(incidents=incs))
        for i in range(5):
            assert f"inc-{i}" in out


# ====================================================================
# display_alert — incident with/without resolution and occurred_at
# ====================================================================

class TestDisplayAlertIncidentDetails:
    def test_incident_with_resolution(self):
        inc = _incident(resolution="restart service")
        out = display_alert(_alert(incidents=[inc]))
        assert "Resolution: restart service" in out

    def test_incident_without_resolution(self):
        inc = _incident(resolution="")
        out = display_alert(_alert(incidents=[inc]))
        assert "Resolution:" not in out

    def test_incident_with_occurred_at(self):
        ts = datetime(2026, 2, 20, 14, 30)
        inc = _incident(occurred_at=ts)
        out = display_alert(_alert(incidents=[inc]))
        assert "When:" in out
        assert "2026-02-20" in out

    def test_incident_without_occurred_at(self):
        inc = _incident(occurred_at=None)
        out = display_alert(_alert(incidents=[inc]))
        assert "When:" not in out


# ====================================================================
# display_alert JSON — all fields present and correct types
# ====================================================================

class TestDisplayAlertJson:
    def test_json_is_valid(self):
        out = display_alert(_alert(), as_json=True)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_json_all_keys_present(self):
        inc = _incident()
        out = display_alert(_alert(incidents=[inc]), as_json=True)
        data = json.loads(out)
        expected_keys = {
            "severity", "anomaly_score", "anomaly_type", "summary",
            "error_count", "warn_count", "total_count", "similar_incidents",
        }
        assert expected_keys == set(data.keys())

    def test_json_types(self):
        inc = _incident()
        out = display_alert(_alert(severity="CRITICAL", score=0.91, incidents=[inc]), as_json=True)
        data = json.loads(out)
        assert isinstance(data["severity"], str)
        assert isinstance(data["anomaly_score"], float)
        assert isinstance(data["anomaly_type"], str)
        assert isinstance(data["summary"], str)
        assert isinstance(data["error_count"], int)
        assert isinstance(data["warn_count"], int)
        assert isinstance(data["total_count"], int)
        assert isinstance(data["similar_incidents"], list)

    def test_json_anomaly_score_rounded(self):
        out = display_alert(_alert(score=0.87654321), as_json=True)
        data = json.loads(out)
        assert data["anomaly_score"] == 0.8765

    def test_json_incident_fields(self):
        ts = datetime(2026, 1, 15, 10, 0, 0)
        inc = _incident(similarity=0.92, label="crash", resolution="fix", occurred_at=ts)
        out = display_alert(_alert(incidents=[inc]), as_json=True)
        data = json.loads(out)
        si = data["similar_incidents"][0]
        assert si["label"] == "crash"
        assert si["resolution"] == "fix"
        assert abs(si["similarity"] - 0.92) < 0.001
        assert si["occurred_at"] is not None

    def test_json_incident_none_occurred_at(self):
        inc = _incident(occurred_at=None)
        out = display_alert(_alert(incidents=[inc]), as_json=True)
        data = json.loads(out)
        assert data["similar_incidents"][0]["occurred_at"] is None


# ====================================================================
# display_scan_report — severity count variations
# ====================================================================

class TestDisplayScanReportCounts:
    def test_zero_alerts(self):
        out = display_scan_report([], total_windows=50)
        assert "No anomalies detected" in out
        assert "Anomalies found: 0" in out

    def test_one_critical_only(self):
        alerts = [_alert(severity="CRITICAL")]
        out = display_scan_report(alerts, total_windows=10)
        assert "CRITICAL: 1" in out
        # WARNING and INFO lines should not appear
        lines = out.split("\n")
        severity_lines = [ln.strip() for ln in lines if ln.strip().startswith(("WARNING:", "INFO:"))]
        assert len(severity_lines) == 0

    def test_mixed_critical_warning_info(self):
        alerts = [
            _alert(severity="CRITICAL"),
            _alert(severity="CRITICAL"),
            _alert(severity="WARNING"),
            _alert(severity="WARNING"),
            _alert(severity="WARNING"),
            _alert(severity="INFO"),
        ]
        out = display_scan_report(alerts, total_windows=200)
        assert "CRITICAL: 2" in out
        assert "WARNING:  3" in out
        assert "INFO:     1" in out
        assert "Anomalies found: 6" in out

    def test_only_info(self):
        alerts = [_alert(severity="INFO")]
        out = display_scan_report(alerts, total_windows=5)
        assert "INFO:     1" in out
        # No CRITICAL or WARNING lines
        assert "CRITICAL:" not in out
        lines_with_warning = [ln for ln in out.split("\n") if "WARNING:" in ln and "Anomalies" not in ln]
        assert len(lines_with_warning) == 0

    def test_scan_report_json_valid(self):
        alerts = [_alert(severity="WARNING")]
        out = display_scan_report(alerts, total_windows=30, as_json=True)
        data = json.loads(out)
        assert data["total_windows"] == 30
        assert data["anomalies_found"] == 1
        assert len(data["alerts"]) == 1

    def test_scan_report_json_empty(self):
        out = display_scan_report([], total_windows=0, as_json=True)
        data = json.loads(out)
        assert data["anomalies_found"] == 0
        assert data["alerts"] == []

    def test_scan_report_json_alert_fields(self):
        alerts = [_alert(severity="CRITICAL", score=0.99, anomaly_type="new_pattern", summary="boom")]
        out = display_scan_report(alerts, total_windows=10, as_json=True)
        data = json.loads(out)
        a = data["alerts"][0]
        assert a["severity"] == "CRITICAL"
        assert abs(a["anomaly_score"] - 0.99) < 0.01
        assert a["anomaly_type"] == "new_pattern"
        assert a["summary"] == "boom"

    def test_scan_report_json_none_time(self):
        w = _window(start=None)
        alerts = [_alert(window=w)]
        out = display_scan_report(alerts, total_windows=1, as_json=True)
        data = json.loads(out)
        assert data["alerts"][0]["time"] is None


# ====================================================================
# display_stats — sources and incidents
# ====================================================================

class TestDisplayStatsRigor:
    def test_zero_sources(self):
        out = display_stats(_index(sources=[]))
        assert "Sources (0):" in out

    def test_one_source(self):
        out = display_stats(_index(sources=["only-svc"]))
        assert "only-svc" in out
        assert "Sources (1):" in out

    def test_many_sources_truncated_at_10(self):
        sources = [f"svc-{i}" for i in range(15)]
        out = display_stats(_index(sources=sources))
        assert "Sources (15):" in out
        assert "and 5 more" in out
        # First 10 shown
        for i in range(10):
            assert f"svc-{i}" in out

    def test_exactly_10_sources_no_truncation(self):
        sources = [f"svc-{i}" for i in range(10)]
        out = display_stats(_index(sources=sources))
        assert "and" not in out.split("Sources")[1].split("=")[0] if "=" in out.split("Sources")[1] else True
        for i in range(10):
            assert f"svc-{i}" in out

    def test_zero_incidents(self):
        out = display_stats(_index(incident_count=0))
        assert "Labeled Incidents" not in out

    def test_one_incident(self):
        out = display_stats(_index(
            incident_count=1,
            incident_labels=["outage"],
            incident_resolutions=["restarted"],
        ))
        assert "Labeled Incidents (1):" in out
        assert "outage" in out
        assert "Resolution: restarted" in out

    def test_many_incidents(self):
        labels = [f"inc-{i}" for i in range(5)]
        resolutions = [f"fix-{i}" for i in range(5)]
        out = display_stats(_index(
            incident_count=5,
            incident_labels=labels,
            incident_resolutions=resolutions,
        ))
        assert "Labeled Incidents (5):" in out
        for i in range(5):
            assert f"inc-{i}" in out

    def test_large_numbers_formatted(self):
        out = display_stats(_index(total_lines=1_000_000, total_windows=50_000))
        assert "1,000,000" in out
        assert "50,000" in out


# ====================================================================
# display_search_results — numbering and percentage
# ====================================================================

class TestDisplaySearchResultsRigor:
    def test_empty_returns_no_matching(self):
        out = display_search_results([])
        assert "No matching incidents found." in out

    def test_numbering_sequential(self):
        results = [_incident(label=f"r-{i}") for i in range(3)]
        out = display_search_results(results)
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_percentage_90(self):
        results = [_incident(similarity=0.90)]
        out = display_search_results(results)
        assert "90%" in out

    def test_percentage_50(self):
        results = [_incident(similarity=0.50)]
        out = display_search_results(results)
        assert "50%" in out

    def test_percentage_1(self):
        results = [_incident(similarity=0.01)]
        out = display_search_results(results)
        assert "1%" in out

    def test_percentage_100(self):
        results = [_incident(similarity=1.0)]
        out = display_search_results(results)
        assert "100%" in out

    def test_count_in_header(self):
        results = [_incident()] * 4
        out = display_search_results(results)
        assert "4 found" in out

    def test_resolution_shown_when_present(self):
        results = [_incident(resolution="scale up")]
        out = display_search_results(results)
        assert "Resolution: scale up" in out

    def test_resolution_hidden_when_empty(self):
        results = [_incident(resolution="")]
        out = display_search_results(results)
        assert "Resolution:" not in out

    def test_occurred_at_shown(self):
        ts = datetime(2026, 3, 15, 9, 0)
        results = [_incident(occurred_at=ts)]
        out = display_search_results(results)
        assert "When:" in out

    def test_occurred_at_hidden(self):
        results = [_incident(occurred_at=None)]
        out = display_search_results(results)
        assert "When:" not in out


# ====================================================================
# Edge cases — long labels, empty strings, special chars
# ====================================================================

class TestEdgeCases:
    def test_very_long_label_1000_chars(self):
        long_label = "X" * 1000
        inc = _incident(label=long_label)
        out = display_alert(_alert(incidents=[inc]))
        assert long_label in out

    def test_empty_summary(self):
        out = display_alert(_alert(summary=""))
        # Should still produce output without crashing
        assert "LogMind Alert" in out

    def test_empty_anomaly_type(self):
        out = display_alert(_alert(anomaly_type=""))
        assert "Type:" in out

    def test_special_chars_in_summary(self):
        out = display_alert(_alert(summary="<script>alert('xss')</script>"))
        assert "<script>" in out  # Not escaped, just raw text

    def test_newlines_in_summary(self):
        out = display_alert(_alert(summary="line1\nline2\nline3"))
        assert "line1\nline2\nline3" in out

    def test_unicode_in_all_fields(self):
        inc = _incident(label="서버 장애", resolution="재시작")
        out = display_alert(_alert(summary="CPU 과부하 발생", incidents=[inc]))
        assert "CPU 과부하 발생" in out
        assert "서버 장애" in out
        assert "재시작" in out


# ====================================================================
# JSON mode consistency — json.loads succeeds for every JSON output
# ====================================================================

class TestJsonConsistency:
    def test_alert_json_parseable_critical(self):
        data = json.loads(display_alert(_alert(severity="CRITICAL", score=0.99), as_json=True))
        assert data["severity"] == "CRITICAL"

    def test_alert_json_parseable_info(self):
        data = json.loads(display_alert(_alert(severity="INFO", score=0.1), as_json=True))
        assert data["severity"] == "INFO"

    def test_alert_json_with_many_incidents(self):
        incs = [_incident(label=f"i-{i}", similarity=0.1 * i) for i in range(5)]
        data = json.loads(display_alert(_alert(incidents=incs), as_json=True))
        assert len(data["similar_incidents"]) == 5

    def test_scan_report_json_parseable(self):
        alerts = [_alert(severity="WARNING"), _alert(severity="CRITICAL")]
        data = json.loads(display_scan_report(alerts, total_windows=99, as_json=True))
        assert data["total_windows"] == 99
        assert data["anomalies_found"] == 2

    def test_alert_json_zero_score(self):
        data = json.loads(display_alert(_alert(score=0.0), as_json=True))
        assert data["anomaly_score"] == 0.0

    def test_alert_json_score_one(self):
        data = json.loads(display_alert(_alert(score=1.0), as_json=True))
        assert data["anomaly_score"] == 1.0

    def test_scan_report_json_large_windows(self):
        data = json.loads(display_scan_report([], total_windows=999_999, as_json=True))
        assert data["total_windows"] == 999_999
