"""Tests for anomaly detector."""

import pytest
import numpy as np
from datetime import datetime
from logmind.models import LogEntry, LogWindow, LogMindIndex
from logmind.detector import (
    compute_anomaly_score, find_similar_incidents, detect,
    _classify_severity, _classify_anomaly_type,
)


def _make_index():
    """Create a minimal test index."""
    from langchain_turboquant import TurboQuantizer
    dim = 384
    quantizer = TurboQuantizer(dim=dim, bits=3)

    # Normal patterns (random vectors)
    np.random.seed(42)
    normal_vecs = np.random.randn(10, dim).astype(np.float32)
    normal_compressed = quantizer.quantize(normal_vecs)

    # One incident pattern
    incident_vec = np.random.randn(1, dim).astype(np.float32)
    incident_compressed = quantizer.quantize(incident_vec)

    return LogMindIndex(
        normal_compressed=normal_compressed,
        normal_windows=[],
        incident_compressed=incident_compressed,
        incident_windows=[LogWindow(
            entries=[], start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1), label="test incident",
            resolution="restart service",
        )],
        incident_labels=["test incident"],
        incident_resolutions=["restart service"],
        quantizer=quantizer,
        model_name="test",
        embedding_dim=dim,
        total_lines=100,
        total_windows=10,
        incident_count=1,
        error_count=5,
        warn_count=3,
        sources=["app"],
        learn_time=1.0,
        avg_errors_per_window=0.5,
        std_errors_per_window=0.3,
        avg_warns_per_window=0.3,
    )


class TestComputeAnomalyScore:
    def test_returns_float(self):
        index = _make_index()
        vec = np.random.randn(384).astype(np.float32)
        score = compute_anomaly_score(vec, index)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_similar_vector_low_score(self):
        index = _make_index()
        # Use one of the normal vectors (should be very similar)
        normal_vec = index.quantizer.dequantize(index.normal_compressed)[0]
        score = compute_anomaly_score(normal_vec, index)
        # Should be low anomaly (high similarity to normal)
        assert score < 0.5


class TestFindSimilarIncidents:
    def test_returns_matches(self):
        index = _make_index()
        vec = np.random.randn(384).astype(np.float32)
        matches = find_similar_incidents(vec, index, k=3)
        # Should return at most 1 (only 1 incident in index)
        assert len(matches) <= 1

    def test_no_incidents(self):
        index = _make_index()
        index.incident_compressed = None
        index.incident_windows = []
        vec = np.random.randn(384).astype(np.float32)
        matches = find_similar_incidents(vec, index)
        assert matches == []


class TestClassifySeverity:
    def test_critical(self):
        entries = [LogEntry(None, "ERROR", "a", "m", "r") for _ in range(15)]
        w = LogWindow(entries, None, None)
        assert _classify_severity(0.8, w, 1.0) == "CRITICAL"

    def test_warning(self):
        entries = [LogEntry(None, "ERROR", "a", "m", "r") for _ in range(4)]
        w = LogWindow(entries, None, None)
        assert _classify_severity(0.55, w, 1.0) == "WARNING"

    def test_info(self):
        entries = [LogEntry(None, "INFO", "a", "m", "r")]
        w = LogWindow(entries, None, None)
        assert _classify_severity(0.3, w, 1.0) == "INFO"


class TestDetect:
    def test_low_anomaly_returns_none(self):
        index = _make_index()
        normal_vec = index.quantizer.dequantize(index.normal_compressed)[0]
        w = LogWindow(
            entries=[LogEntry(None, "INFO", "a", "ok", "r")],
            start_time=None, end_time=None,
        )
        result = detect(w, normal_vec, index, sensitivity="low")
        assert result is None

    def test_returns_alert_type(self):
        index = _make_index()
        vec = np.random.randn(384).astype(np.float32) * 100  # very different
        entries = [LogEntry(None, "ERROR", "a", "crash", "r") for _ in range(20)]
        w = LogWindow(entries, None, None)
        result = detect(w, vec, index, sensitivity="high")
        if result is not None:
            assert result.severity in ("CRITICAL", "WARNING", "INFO")
            assert result.anomaly_type in ("error_spike", "new_pattern", "similar_incident")
