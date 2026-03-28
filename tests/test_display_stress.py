"""Stress tests for logmind.display — 30+ tests covering
display_alert, display_scan_report, display_stats, and display_search_results."""

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


def _make_entry(
    ts: datetime | None,
    level: str = "INFO",
    source: str = "app",
    message: str = "ok",
) -> LogEntry:
    raw = f"[{level}] {source}: {message}"
    return LogEntry(timestamp=ts, level=level, source=source, message=message, raw=raw)


def _make_window(
    entries: List[LogEntry] | None = None,
    start: datetime | None = BASE_TIME,
    end: datetime | None = None,
    label: str = "",
    resolution: str = "",
) -> LogWindow:
    if entries is None:
        entries = [_make_entry(start)]
    if end is None:
        end = start + timedelta(seconds=60) if start else None
    return LogWindow(entries=entries, start_time=start, end_time=end, label=label, resolution=resolution)


def _make_alert(
    severity: str = "WARNING",
    score: float = 0.75,
    anomaly_type: str = "error_spike",
    summary: str = "Error rate spiked to 42/min",
    incidents: List[IncidentMatch] | None = None,
    window: LogWindow | None = None,
) -> AnomalyAlert:
    if window is None:
        window = _make_window(entries=[
            _make_entry(BASE_TIME, "ERROR", "svc", "fail"),
            _make_entry(BASE_TIME, "WARN", "svc", "slow"),
            _make_entry(BASE_TIME, "INFO", "svc", "ok"),
        ])
    return AnomalyAlert(
        current_window=window,
        anomaly_score=score,
        anomaly_type=anomaly_type,
        similar_incidents=incidents or [],
        severity=severity,
        summary=summary,
    )


def _make_incident_match(
    similarity: float = 0.85,
    label: str = "disk-full",
    resolution: str = "expanded volume",
    occurred_at: datetime | None = BASE_TIME - timedelta(days=7),
) -> IncidentMatch:
    return IncidentMatch(
        similarity=similarity,
        label=label,
        resolution=resolution,
        occurred_at=occurred_at,
    )


def _make_index(**overrides) -> LogMindIndex:
    q = TurboQuantizer(dim=DIM, bits=BITS)
    defaults = dict(
        normal_compressed=None,
        normal_windows=[],
        incident_compressed=None,
        incident_windows=[],
        incident_labels=[],
        incident_resolutions=[],
        quantizer=q,
        model_name="all-MiniLM-L6-v2",
        embedding_dim=DIM,
        total_lines=1000,
        total_windows=50,
        incident_count=0,
        error_count=120,
        warn_count=45,
        sources=["svc-a", "svc-b"],
        learn_time=3.5,
    )
    defaults.update(overrides)
    return LogMindIndex(**defaults)


# ====================================================================
# _severity_color / _severity_icon helpers
# ====================================================================

class TestSeverityHelpers:
    def test_color_critical(self):
        assert _severity_color("CRITICAL") == "red"

    def test_color_warning(self):
        assert _severity_color("WARNING") == "yellow"

    def test_color_info(self):
        assert _severity_color("INFO") == "blue"

    def test_color_unknown(self):
        assert _severity_color("UNKNOWN") == "white"

    def test_icon_critical(self):
        assert _severity_icon("CRITICAL") == "[!!!]"

    def test_icon_warning(self):
        assert _severity_icon("WARNING") == "[!!]"

    def test_icon_info(self):
        assert _severity_icon("INFO") == "[i]"

    def test_icon_unknown(self):
        assert _severity_icon("WHATEVER") == "[?]"


# ====================================================================
# display_alert
# ====================================================================

class TestDisplayAlert:
    def test_critical_severity_text(self):
        alert = _make_alert(severity="CRITICAL", score=0.99)
        output = display_alert(alert)
        assert "[!!!]" in output
        assert "CRITICAL" in output

    def test_warning_severity_text(self):
        alert = _make_alert(severity="WARNING")
        output = display_alert(alert)
        assert "[!!]" in output
        assert "WARNING" in output

    def test_info_severity_text(self):
        alert = _make_alert(severity="INFO", score=0.3)
        output = display_alert(alert)
        assert "[i]" in output
        assert "INFO" in output

    def test_anomaly_score_formatted(self):
        alert = _make_alert(score=0.8765)
        output = display_alert(alert)
        assert "0.88" in output

    def test_anomaly_type_shown(self):
        alert = _make_alert(anomaly_type="new_pattern")
        output = display_alert(alert)
        assert "new_pattern" in output

    def test_summary_included(self):
        alert = _make_alert(summary="CPU usage at 100%")
        output = display_alert(alert)
        assert "CPU usage at 100%" in output

    def test_time_shown_when_present(self):
        alert = _make_alert()
        output = display_alert(alert)
        assert "Time:" in output

    def test_time_hidden_when_none(self):
        window = _make_window(start=None, end=None)
        alert = _make_alert(window=window)
        output = display_alert(alert)
        assert "Time:" not in output

    def test_no_incidents_section_when_empty(self):
        alert = _make_alert(incidents=[])
        output = display_alert(alert)
        assert "Similar Past Incidents:" not in output

    def test_with_single_incident(self):
        inc = _make_incident_match(similarity=0.92, label="oom-kill", resolution="restart")
        alert = _make_alert(incidents=[inc])
        output = display_alert(alert)
        assert "Similar Past Incidents:" in output
        assert "oom-kill" in output
        assert "Resolution: restart" in output

    def test_with_multiple_incidents(self):
        incs = [
            _make_incident_match(similarity=0.9, label="inc-1", resolution="fix-1"),
            _make_incident_match(similarity=0.7, label="inc-2", resolution="fix-2"),
        ]
        alert = _make_alert(incidents=incs)
        output = display_alert(alert)
        assert "inc-1" in output
        assert "inc-2" in output

    def test_incident_without_resolution(self):
        inc = _make_incident_match(resolution="")
        alert = _make_alert(incidents=[inc])
        output = display_alert(alert)
        assert "Resolution:" not in output

    def test_incident_without_occurred_at(self):
        inc = _make_incident_match(occurred_at=None)
        alert = _make_alert(incidents=[inc])
        output = display_alert(alert)
        assert "When:" not in output

    def test_separator_lines(self):
        alert = _make_alert()
        output = display_alert(alert)
        assert "=" * 60 in output
        assert "-" * 60 in output

    def test_json_mode_valid_json(self):
        alert = _make_alert()
        output = display_alert(alert, as_json=True)
        data = json.loads(output)
        assert "severity" in data
        assert "anomaly_score" in data

    def test_json_mode_fields(self):
        inc = _make_incident_match()
        alert = _make_alert(
            severity="CRITICAL", score=0.95, anomaly_type="error_spike",
            summary="big problem", incidents=[inc],
        )
        output = display_alert(alert, as_json=True)
        data = json.loads(output)
        assert data["severity"] == "CRITICAL"
        assert data["anomaly_type"] == "error_spike"
        assert abs(data["anomaly_score"] - 0.95) < 0.01
        assert data["summary"] == "big problem"
        assert len(data["similar_incidents"]) == 1
        assert data["similar_incidents"][0]["label"] == "disk-full"

    def test_json_mode_counts(self):
        window = _make_window(entries=[
            _make_entry(BASE_TIME, "ERROR", "svc", "e1"),
            _make_entry(BASE_TIME, "ERROR", "svc", "e2"),
            _make_entry(BASE_TIME, "WARN", "svc", "w1"),
            _make_entry(BASE_TIME, "INFO", "svc", "i1"),
        ])
        alert = _make_alert(window=window)
        data = json.loads(display_alert(alert, as_json=True))
        assert data["error_count"] == 2
        assert data["warn_count"] == 1
        assert data["total_count"] == 4

    def test_long_summary(self):
        long_summary = "A" * 500
        alert = _make_alert(summary=long_summary)
        output = display_alert(alert)
        assert long_summary in output

    def test_unicode_in_summary(self):
        alert = _make_alert(summary="서버 오류 발생! メモリ不足 🔥")
        output = display_alert(alert)
        assert "서버 오류 발생" in output


# ====================================================================
# display_scan_report
# ====================================================================

class TestDisplayScanReport:
    def test_empty_alerts(self):
        output = display_scan_report([], total_windows=100)
        assert "No anomalies detected" in output
        assert "Windows analyzed: 100" in output

    def test_single_alert(self):
        alert = _make_alert(severity="WARNING")
        output = display_scan_report([alert], total_windows=50)
        assert "Anomalies found: 1" in output
        assert "WARNING:" in output

    def test_multiple_mixed_severities(self):
        alerts = [
            _make_alert(severity="CRITICAL"),
            _make_alert(severity="CRITICAL"),
            _make_alert(severity="WARNING"),
            _make_alert(severity="INFO"),
        ]
        output = display_scan_report(alerts, total_windows=200)
        assert "CRITICAL: 2" in output
        assert "WARNING:  1" in output
        assert "INFO:     1" in output
        assert "Anomalies found: 4" in output

    def test_only_critical(self):
        alerts = [_make_alert(severity="CRITICAL")]
        output = display_scan_report(alerts, total_windows=10)
        assert "CRITICAL: 1" in output
        assert "WARNING:" not in output.split("CRITICAL:")[0]  # no WARNING line before

    def test_json_mode_empty(self):
        output = display_scan_report([], total_windows=10, as_json=True)
        data = json.loads(output)
        assert data["total_windows"] == 10
        assert data["anomalies_found"] == 0
        assert data["alerts"] == []

    def test_json_mode_with_alerts(self):
        alerts = [_make_alert(severity="WARNING"), _make_alert(severity="CRITICAL")]
        output = display_scan_report(alerts, total_windows=30, as_json=True)
        data = json.loads(output)
        assert data["anomalies_found"] == 2
        assert len(data["alerts"]) == 2

    def test_report_header(self):
        output = display_scan_report([], total_windows=0)
        assert "LogMind Scan Report" in output

    def test_zero_windows(self):
        output = display_scan_report([], total_windows=0)
        assert "Windows analyzed: 0" in output


# ====================================================================
# display_stats
# ====================================================================

class TestDisplayStats:
    def test_basic_stats(self):
        index = _make_index()
        output = display_stats(index)
        assert "Total log lines:" in output
        assert "1,000" in output
        assert "Total windows:" in output

    def test_error_and_warn_counts(self):
        index = _make_index(error_count=500, warn_count=200)
        output = display_stats(index)
        assert "500" in output
        assert "200" in output

    def test_model_name_shown(self):
        index = _make_index(model_name="custom-model-xyz")
        output = display_stats(index)
        assert "custom-model-xyz" in output

    def test_embedding_dim_shown(self):
        index = _make_index(embedding_dim=DIM)
        output = display_stats(index)
        assert str(DIM) in output

    def test_learn_time(self):
        index = _make_index(learn_time=12.3)
        output = display_stats(index)
        assert "12.3s" in output

    def test_sources_listed(self):
        index = _make_index(sources=["web-server", "db-primary", "cache"])
        output = display_stats(index)
        assert "web-server" in output
        assert "db-primary" in output
        assert "cache" in output

    def test_sources_truncated_at_10(self):
        sources = [f"svc-{i}" for i in range(15)]
        index = _make_index(sources=sources)
        output = display_stats(index)
        assert "and 5 more" in output

    def test_no_incidents(self):
        index = _make_index(incident_count=0)
        output = display_stats(index)
        assert "Labeled Incidents" not in output

    def test_with_incidents(self):
        index = _make_index(
            incident_count=2,
            incident_labels=["disk-full", "oom-kill"],
            incident_resolutions=["expand disk", "increase heap"],
        )
        output = display_stats(index)
        assert "Labeled Incidents (2):" in output
        assert "disk-full" in output
        assert "oom-kill" in output
        assert "Resolution: expand disk" in output

    def test_incident_without_resolution(self):
        index = _make_index(
            incident_count=1,
            incident_labels=["mystery"],
            incident_resolutions=[""],
        )
        output = display_stats(index)
        assert "mystery" in output
        # Empty resolution should not print "Resolution:"
        lines = output.split("\n")
        mystery_line_idx = next(i for i, l in enumerate(lines) if "mystery" in l)
        if mystery_line_idx + 1 < len(lines):
            assert "Resolution:" not in lines[mystery_line_idx + 1]

    def test_zero_everything(self):
        index = _make_index(
            total_lines=0, total_windows=0, error_count=0, warn_count=0,
            sources=[], learn_time=0.0, incident_count=0,
        )
        output = display_stats(index)
        assert "Total log lines:" in output

    def test_avg_errors_per_window(self):
        index = _make_index(avg_errors_per_window=4.5)
        output = display_stats(index)
        assert "4.5" in output


# ====================================================================
# display_search_results
# ====================================================================

class TestDisplaySearchResults:
    def test_empty_results(self):
        output = display_search_results([])
        assert "No matching incidents found." in output

    def test_single_result(self):
        results = [_make_incident_match(similarity=0.91, label="cpu-spike", resolution="scale up")]
        output = display_search_results(results)
        assert "1 found" in output
        assert "cpu-spike" in output
        assert "91%" in output
        assert "Resolution: scale up" in output

    def test_multiple_results(self):
        results = [
            _make_incident_match(similarity=0.95, label="inc-a"),
            _make_incident_match(similarity=0.80, label="inc-b"),
            _make_incident_match(similarity=0.60, label="inc-c"),
        ]
        output = display_search_results(results)
        assert "3 found" in output
        assert "inc-a" in output
        assert "inc-b" in output
        assert "inc-c" in output

    def test_result_numbering(self):
        results = [
            _make_incident_match(label="first"),
            _make_incident_match(label="second"),
        ]
        output = display_search_results(results)
        assert "[1]" in output
        assert "[2]" in output

    def test_result_without_resolution(self):
        results = [_make_incident_match(resolution="")]
        output = display_search_results(results)
        assert "Resolution:" not in output

    def test_result_without_occurred_at(self):
        results = [_make_incident_match(occurred_at=None)]
        output = display_search_results(results)
        assert "When:" not in output

    def test_result_with_occurred_at(self):
        ts = datetime(2026, 2, 15, 8, 30, 0)
        results = [_make_incident_match(occurred_at=ts)]
        output = display_search_results(results)
        assert "When:" in output
        assert "2026-02-15" in output

    def test_long_label(self):
        long_label = "A" * 200
        results = [_make_incident_match(label=long_label)]
        output = display_search_results(results)
        assert long_label in output

    def test_unicode_label(self):
        results = [_make_incident_match(label="서버 장애 - メモリ不足")]
        output = display_search_results(results)
        assert "서버 장애" in output

    def test_header_present(self):
        results = [_make_incident_match()]
        output = display_search_results(results)
        assert "LogMind Search Results" in output
        assert "=" * 60 in output


# ====================================================================
# Edge cases
# ====================================================================

class TestEdgeCases:
    def test_window_with_no_entries(self):
        window = _make_window(entries=[])
        alert = _make_alert(window=window)
        data = json.loads(display_alert(alert, as_json=True))
        assert data["error_count"] == 0
        assert data["total_count"] == 0

    def test_none_start_time_in_window(self):
        window = _make_window(start=None)
        alert = _make_alert(window=window)
        output = display_alert(alert)
        assert "Time:" not in output

    def test_json_alert_none_occurred_at(self):
        inc = _make_incident_match(occurred_at=None)
        alert = _make_alert(incidents=[inc])
        data = json.loads(display_alert(alert, as_json=True))
        assert data["similar_incidents"][0]["occurred_at"] is None

    def test_scan_report_json_none_time(self):
        window = _make_window(start=None)
        alert = _make_alert(window=window)
        data = json.loads(display_scan_report([alert], total_windows=1, as_json=True))
        assert data["alerts"][0]["time"] is None

    def test_similarity_percentage_formatting(self):
        inc = _make_incident_match(similarity=0.123)
        alert = _make_alert(incidents=[inc])
        output = display_alert(alert)
        assert "12%" in output

    def test_display_alert_returns_string(self):
        alert = _make_alert()
        assert isinstance(display_alert(alert), str)
        assert isinstance(display_alert(alert, as_json=True), str)

    def test_display_scan_report_returns_string(self):
        assert isinstance(display_scan_report([], 0), str)
        assert isinstance(display_scan_report([], 0, as_json=True), str)
