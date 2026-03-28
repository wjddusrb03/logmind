"""Tests for display formatting."""

import json
import pytest
from datetime import datetime
from logmind.models import (
    AnomalyAlert, IncidentMatch, LogEntry, LogMindIndex, LogWindow,
)
from logmind.display import (
    display_alert, display_scan_report, display_stats, display_search_results,
)


def _make_alert(severity="WARNING", score=0.6):
    entries = [LogEntry(None, "ERROR", "DB", "connection failed", "raw")]
    window = LogWindow(entries, datetime(2024,1,1,10,30), datetime(2024,1,1,10,31))
    incidents = [IncidentMatch(0.85, "DB outage", "restart", datetime(2024,1,1))]
    return AnomalyAlert(window, score, "similar_incident", incidents, severity, "Test alert")


class TestDisplayAlert:
    def test_text_output(self):
        alert = _make_alert()
        output = display_alert(alert)
        assert "WARNING" in output
        assert "Test alert" in output
        assert "DB outage" in output

    def test_json_output(self):
        alert = _make_alert()
        output = display_alert(alert, as_json=True)
        data = json.loads(output)
        assert data["severity"] == "WARNING"
        assert len(data["similar_incidents"]) == 1

    def test_critical_icon(self):
        alert = _make_alert(severity="CRITICAL")
        output = display_alert(alert)
        assert "[!!!]" in output


class TestDisplayScanReport:
    def test_no_anomalies(self):
        output = display_scan_report([], 10)
        assert "No anomalies detected" in output

    def test_with_anomalies(self):
        alerts = [_make_alert(), _make_alert(severity="CRITICAL")]
        output = display_scan_report(alerts, 10)
        assert "Anomalies found: 2" in output

    def test_json_report(self):
        alerts = [_make_alert()]
        output = display_scan_report(alerts, 5, as_json=True)
        data = json.loads(output)
        assert data["total_windows"] == 5
        assert data["anomalies_found"] == 1


class TestDisplayStats:
    def test_basic_stats(self):
        from langchain_turboquant import TurboQuantizer
        import numpy as np
        q = TurboQuantizer(dim=32, bits=3)
        index = LogMindIndex(
            normal_compressed=q.quantize(np.random.randn(5, 32).astype(np.float32)),
            normal_windows=[], incident_compressed=None,
            incident_windows=[], incident_labels=[], incident_resolutions=[],
            quantizer=q, model_name="test", embedding_dim=32,
            total_lines=1000, total_windows=50, incident_count=0,
            error_count=25, warn_count=10, sources=["api", "db"],
            learn_time=2.0,
        )
        output = display_stats(index)
        assert "1,000" in output
        assert "api" in output


class TestDisplaySearchResults:
    def test_no_results(self):
        output = display_search_results([])
        assert "No matching" in output

    def test_with_results(self):
        results = [
            IncidentMatch(0.9, "outage", "restart", datetime(2024,1,1)),
            IncidentMatch(0.7, "slowdown", "scale up", datetime(2024,2,1)),
        ]
        output = display_search_results(results)
        assert "outage" in output
        assert "restart" in output
        assert "2 found" in output
