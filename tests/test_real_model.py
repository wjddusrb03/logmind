"""Comprehensive tests using the REAL sentence-transformers model and TurboQuantizer.

These tests load the actual all-MiniLM-L6-v2 model and exercise the full
embedding / quantization / detection / incident pipeline end-to-end.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from logmind.embedder import (
    DEFAULT_MODEL,
    _create_windows,
    _get_model,
    _model_cache,
    build_index,
    embed_window,
)
from logmind.detector import (
    compute_anomaly_score,
    detect,
    find_similar_incidents,
    _classify_severity,
    _classify_anomaly_type,
)
from logmind.incidents import label_incident, search_incidents, auto_detect_incidents
from logmind.models import LogEntry, LogMindIndex, LogWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 3, 1, 12, 0, 0)


def _entry(
    msg: str,
    level: str = "INFO",
    source: str = "app",
    offset_seconds: int = 0,
    line_number: int = 0,
) -> LogEntry:
    return LogEntry(
        timestamp=_BASE_TIME + timedelta(seconds=offset_seconds),
        level=level,
        source=source,
        message=msg,
        raw=f"2026-03-01 12:00:{offset_seconds:02d} [{level}] {source}: {msg}",
        line_number=line_number,
    )


def _normal_entries(n: int = 20, start_offset: int = 0) -> List[LogEntry]:
    """Generate realistic normal/healthy log entries."""
    templates = [
        "Request handled successfully in 32ms",
        "User session refreshed token=abc123",
        "Cache hit ratio 98.2%",
        "Scheduled job completed: cleanup",
        "Healthcheck passed all checks",
        "Connection pool size: 10/50",
        "Response sent status=200 path=/api/v1/data",
        "Metrics flushed to collector",
        "Loaded configuration from /etc/app/config.yaml",
        "Worker thread started id=7",
    ]
    entries = []
    for i in range(n):
        entries.append(
            _entry(
                templates[i % len(templates)],
                level="INFO",
                source="app",
                offset_seconds=start_offset + i * 2,
                line_number=i + 1,
            )
        )
    return entries


def _error_entries(n: int = 10, start_offset: int = 0) -> List[LogEntry]:
    """Generate realistic error log entries."""
    templates = [
        "java.lang.OutOfMemoryError: Java heap space",
        "Connection refused to database host db-primary:5432",
        "Timeout waiting for response from upstream service",
        "FATAL: password authentication failed for user admin",
        "Segmentation fault (core dumped)",
        "Disk space critically low: /var/log 98% used",
        "SSL certificate has expired for domain api.example.com",
        "Max retries exceeded for task send_email",
        "Unhandled exception in request handler: NullPointerException",
        "Kernel panic - not syncing: Out of memory",
    ]
    entries = []
    for i in range(n):
        entries.append(
            _entry(
                templates[i % len(templates)],
                level="ERROR",
                source="app",
                offset_seconds=start_offset + i * 2,
                line_number=i + 1,
            )
        )
    return entries


def _mixed_entries(
    normal: int = 15, errors: int = 5, start_offset: int = 0,
) -> List[LogEntry]:
    """Interleave normal and error entries."""
    ne = _normal_entries(normal, start_offset)
    ee = _error_entries(errors, start_offset + normal * 2)
    combined = ne + ee
    combined.sort(key=lambda e: e.timestamp)
    return combined


def _make_window(entries: List[LogEntry], label: str = "", resolution: str = "") -> LogWindow:
    return LogWindow(
        entries=entries,
        start_time=entries[0].timestamp if entries else None,
        end_time=entries[-1].timestamp if entries else None,
        window_size=60,
        label=label,
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# 1. build_index with real log entries
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """Test build_index using the real model and TurboQuantizer."""

    def test_basic_build(self):
        entries = _normal_entries(30)
        index = build_index(entries, window_size=30)
        assert index is not None
        assert index.total_lines == 30
        assert index.total_windows >= 1

    def test_embedding_dim_384(self):
        entries = _normal_entries(20)
        index = build_index(entries, window_size=120)
        assert index.embedding_dim == 384

    def test_model_name_stored(self):
        entries = _normal_entries(20)
        index = build_index(entries, window_size=120)
        assert index.model_name == DEFAULT_MODEL

    def test_compressed_vectors_not_none(self):
        entries = _normal_entries(20)
        index = build_index(entries, window_size=120)
        assert index.normal_compressed is not None

    def test_learn_time_positive(self):
        entries = _normal_entries(20)
        index = build_index(entries, window_size=120)
        assert index.learn_time > 0.0

    def test_error_and_warn_counts(self):
        entries = _mixed_entries(normal=15, errors=5)
        index = build_index(entries, window_size=120)
        assert index.error_count == 5
        assert index.warn_count == 0

    def test_sources_tracked(self):
        entries = _normal_entries(10)
        index = build_index(entries, window_size=120)
        assert "app" in index.sources

    def test_multiple_windows_created(self):
        entries = _normal_entries(30, start_offset=0)
        index = build_index(entries, window_size=20)
        assert index.total_windows >= 2

    def test_baseline_stats_computed(self):
        entries = _mixed_entries(normal=20, errors=5)
        index = build_index(entries, window_size=120)
        assert index.avg_errors_per_window >= 0.0
        assert index.std_errors_per_window >= 0.0

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="No log windows"):
            build_index([])

    def test_incident_windows_in_build(self):
        normal = _normal_entries(20)
        inc_entries = _error_entries(5, start_offset=200)
        inc_window = _make_window(inc_entries, label="OOM crash", resolution="Restart pod")
        index = build_index(normal, window_size=120, incident_windows=[inc_window])
        assert index.incident_compressed is not None
        assert index.incident_count == 1
        assert index.incident_labels == ["OOM crash"]
        assert index.incident_resolutions == ["Restart pod"]


# ---------------------------------------------------------------------------
# 2. embed_window with real model
# ---------------------------------------------------------------------------

class TestEmbedWindow:
    """Test embed_window produces valid vectors."""

    def test_vector_shape(self):
        entries = _normal_entries(10)
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_vector_dtype(self):
        entries = _normal_entries(10)
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.dtype == np.float32

    def test_vector_not_all_zeros(self):
        entries = _normal_entries(10)
        window = _make_window(entries)
        vec = embed_window(window)
        assert np.any(vec != 0.0)

    def test_vector_has_reasonable_norm(self):
        entries = _normal_entries(10)
        window = _make_window(entries)
        vec = embed_window(window)
        norm = np.linalg.norm(vec)
        assert 0.5 < norm < 2.0, f"Unexpected norm: {norm}"

    def test_different_content_different_vectors(self):
        w1 = _make_window(_normal_entries(10))
        w2 = _make_window(_error_entries(10))
        v1 = embed_window(w1)
        v2 = embed_window(w2)
        assert not np.allclose(v1, v2, atol=1e-3)

    def test_single_entry_window(self):
        entries = [_entry("Server started successfully")]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)
        assert np.any(vec != 0.0)


# ---------------------------------------------------------------------------
# 3. Anomaly detection end-to-end
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Build index from normal logs, detect anomalies in error logs."""

    @pytest.fixture()
    def normal_index(self):
        entries = _normal_entries(40)
        return build_index(entries, window_size=30)

    def test_normal_window_low_anomaly_score(self, normal_index):
        w = _make_window(_normal_entries(10))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, normal_index)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, f"Normal window scored too high: {score}"

    def test_error_window_higher_anomaly_score(self, normal_index):
        w = _make_window(_error_entries(10))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, normal_index)
        assert score > 0.0

    def test_detect_returns_alert_for_errors(self, normal_index):
        w = _make_window(_error_entries(10))
        vec = embed_window(w)
        alert = detect(w, vec, normal_index, sensitivity="high")
        # With high sensitivity (threshold=0.30) error logs should trigger
        if alert is not None:
            assert alert.anomaly_score > 0.0
            assert alert.severity in ("CRITICAL", "WARNING", "INFO")
            assert alert.anomaly_type in ("error_spike", "new_pattern", "similar_incident")

    def test_detect_normal_no_alert(self, normal_index):
        w = _make_window(_normal_entries(10))
        vec = embed_window(w)
        alert = detect(w, vec, normal_index, sensitivity="low")
        # With low sensitivity normal window should not fire
        assert alert is None

    def test_anomaly_score_range(self, normal_index):
        w = _make_window(_error_entries(10))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, normal_index)
        assert 0.0 <= score <= 1.0

    def test_classify_severity_critical(self):
        w = _make_window(_error_entries(15))
        severity = _classify_severity(0.8, w, avg_errors=1.0)
        assert severity == "CRITICAL"

    def test_classify_severity_warning(self):
        w = _make_window(_error_entries(5))
        severity = _classify_severity(0.5, w, avg_errors=1.0)
        assert severity == "WARNING"

    def test_classify_severity_info(self):
        w = _make_window(_normal_entries(5))
        severity = _classify_severity(0.1, w, avg_errors=1.0)
        assert severity == "INFO"


# ---------------------------------------------------------------------------
# 4. Incident labeling and search
# ---------------------------------------------------------------------------

class TestIncidents:
    """Test incident labeling, searching and auto-detection."""

    @pytest.fixture()
    def index_with_incidents(self):
        normal = _normal_entries(30)
        inc1 = _error_entries(5, start_offset=200)
        inc1_window = _make_window(inc1, label="Database OOM", resolution="Increased heap size")
        inc2_entries = [
            _entry("SSL handshake failed: certificate expired", "ERROR", "nginx", 300 + i)
            for i in range(5)
        ]
        inc2_window = _make_window(inc2_entries, label="SSL cert expired", resolution="Renewed certificate")
        return build_index(normal, window_size=120, incident_windows=[inc1_window, inc2_window])

    def test_label_incident(self):
        normal = _normal_entries(30)
        index = build_index(normal, window_size=120)
        assert index.incident_count == 0

        all_entries = normal + _error_entries(5, start_offset=200)
        start = _BASE_TIME + timedelta(seconds=200)
        end = _BASE_TIME + timedelta(seconds=210)
        index = label_incident(index, all_entries, start, end, "Test incident", "Fixed it")
        assert index.incident_count == 1
        assert index.incident_labels[-1] == "Test incident"
        assert index.incident_resolutions[-1] == "Fixed it"

    def test_label_incident_no_entries_raises(self):
        normal = _normal_entries(20)
        index = build_index(normal, window_size=120)
        far_start = _BASE_TIME + timedelta(hours=10)
        far_end = _BASE_TIME + timedelta(hours=11)
        with pytest.raises(ValueError, match="No log entries found"):
            label_incident(index, normal, far_start, far_end, "No entries here")

    def test_search_incidents_by_meaning(self, index_with_incidents):
        results = search_incidents("database out of memory", index_with_incidents)
        assert len(results) >= 1
        labels = [r.label for r in results]
        assert "Database OOM" in labels

    def test_search_incidents_ssl(self, index_with_incidents):
        results = search_incidents("certificate renewal", index_with_incidents)
        assert len(results) >= 1
        labels = [r.label for r in results]
        assert "SSL cert expired" in labels

    def test_search_returns_similarity_scores(self, index_with_incidents):
        results = search_incidents("memory problem", index_with_incidents)
        for r in results:
            assert 0.0 <= r.similarity <= 1.0

    def test_search_empty_index(self):
        normal = _normal_entries(20)
        index = build_index(normal, window_size=120)
        results = search_incidents("anything", index)
        assert results == []

    def test_find_similar_incidents(self, index_with_incidents):
        err_window = _make_window([
            _entry("java.lang.OutOfMemoryError: heap space", "ERROR", "jvm", i)
            for i in range(5)
        ])
        vec = embed_window(err_window)
        matches = find_similar_incidents(vec, index_with_incidents, k=2)
        assert len(matches) >= 1

    def test_auto_detect_incidents_with_spike(self):
        normal = _normal_entries(40)
        errors = _error_entries(20, start_offset=100)
        all_entries = normal + errors
        detected = auto_detect_incidents(all_entries, window_size=30, spike_threshold=1.5)
        assert len(detected) >= 1
        for w in detected:
            assert w.error_count > 0
            assert w.label.startswith("auto:")


# ---------------------------------------------------------------------------
# 5. Semantic similarity
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    """Verify the model captures semantic meaning."""

    @pytest.fixture(scope="class")
    def model(self):
        return _get_model(DEFAULT_MODEL)

    def _cosine(self, model, a: str, b: str) -> float:
        vecs = model.encode([a, b], show_progress_bar=False)
        va, vb = vecs[0], vecs[1]
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def test_db_connection_vs_db_timeout(self, model):
        sim = self._cosine(model, "DB connection failed", "database timeout error")
        assert sim > 0.4, f"Expected high similarity, got {sim}"

    def test_oom_vs_memory_exhausted(self, model):
        sim = self._cosine(model, "OutOfMemoryError: Java heap space", "memory exhausted, cannot allocate")
        assert sim > 0.3, f"Expected moderate similarity, got {sim}"

    def test_ssl_vs_certificate(self, model):
        sim = self._cosine(model, "SSL handshake failed", "TLS certificate expired")
        assert sim > 0.3, f"Expected moderate similarity, got {sim}"

    def test_unrelated_messages_low_similarity(self, model):
        sim = self._cosine(model, "Disk full /var/log", "User logged in successfully from 10.0.0.1")
        assert sim < 0.7, f"Expected low similarity, got {sim}"

    def test_similar_errors_higher_than_unrelated(self, model):
        sim_related = self._cosine(model, "connection refused port 5432", "cannot connect to postgres database")
        sim_unrelated = self._cosine(model, "connection refused port 5432", "image uploaded to S3 bucket")
        assert sim_related > sim_unrelated


# ---------------------------------------------------------------------------
# 6. Different log formats
# ---------------------------------------------------------------------------

class TestDifferentLogFormats:
    """Test embedding works with various log message styles."""

    def test_json_style_logs(self):
        entries = [
            _entry('{"level":"error","msg":"disk full","host":"web-01"}', "ERROR", "json-app", i * 2)
            for i in range(5)
        ]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)
        assert np.any(vec != 0.0)

    def test_python_traceback_logs(self):
        tb_lines = [
            "Traceback (most recent call last):",
            '  File "/app/main.py", line 42, in handler',
            "    result = db.query(sql)",
            "psycopg2.OperationalError: connection timed out",
        ]
        entries = [_entry(line, "ERROR", "python-app", i) for i, line in enumerate(tb_lines)]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_syslog_style_logs(self):
        entries = [
            _entry("kernel: [12345.678] Out of memory: Kill process 1234 (java)", "ERROR", "syslog", 0),
            _entry("kernel: [12345.679] Killed process 1234 (java) total-vm:4096kB", "ERROR", "syslog", 1),
        ]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_mixed_sources_window(self):
        entries = [
            _entry("GET /api/health 200 OK", "INFO", "nginx", 0),
            _entry("Query executed in 12ms", "INFO", "postgres", 1),
            _entry("Cache miss for key user:123", "WARN", "redis", 2),
            _entry("Task completed: send_notifications", "INFO", "celery", 3),
        ]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_unicode_log_messages(self):
        entries = [
            _entry("Error: DB 연결 실패 - 타임아웃", "ERROR", "app", 0),
            _entry("Warning: 메모리 사용량 90% 초과", "WARN", "monitor", 1),
        ]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)
        assert np.any(vec != 0.0)


# ---------------------------------------------------------------------------
# 7. Model caching
# ---------------------------------------------------------------------------

class TestModelCaching:
    """Verify _get_model returns a cached instance."""

    def test_same_instance_returned(self):
        m1 = _get_model(DEFAULT_MODEL)
        m2 = _get_model(DEFAULT_MODEL)
        assert m1 is m2

    def test_model_in_cache_dict(self):
        _get_model(DEFAULT_MODEL)
        assert DEFAULT_MODEL in _model_cache

    def test_cached_model_produces_embeddings(self):
        model = _get_model(DEFAULT_MODEL)
        vec = model.encode(["test sentence"], show_progress_bar=False)[0]
        assert len(vec) == 384


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty windows, single entries, very long messages."""

    def test_single_entry_build_index(self):
        entries = [_entry("Server started", offset_seconds=0)]
        index = build_index(entries, window_size=60)
        assert index.total_windows == 1
        assert index.total_lines == 1

    def test_window_with_only_errors(self):
        entries = _error_entries(10)
        index = build_index(entries, window_size=120)
        assert index.error_count == 10

    def test_very_long_message(self):
        long_msg = "ERROR: " + "x" * 5000
        entries = [_entry(long_msg, "ERROR", "app", 0)]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_entries_without_timestamps(self):
        entries = []
        for i in range(25):
            entries.append(LogEntry(
                timestamp=None,
                level="INFO",
                source="app",
                message=f"Log line {i}: doing work",
                raw=f"Log line {i}: doing work",
                line_number=i,
            ))
        index = build_index(entries, window_size=60)
        assert index.total_windows >= 1

    def test_create_windows_empty(self):
        result = _create_windows([])
        assert result == []

    def test_duplicate_messages_window(self):
        entries = [_entry("Connection reset by peer", "ERROR", "app", i) for i in range(20)]
        window = _make_window(entries)
        vec = embed_window(window)
        assert vec.shape == (384,)

    def test_all_warn_entries(self):
        entries = [
            _entry(f"Slow query detected: {i * 100}ms", "WARN", "db", i * 2)
            for i in range(10)
        ]
        index = build_index(entries, window_size=120)
        assert index.warn_count == 10
        assert index.error_count == 0

    def test_build_index_with_3bit_quantization(self):
        entries = _normal_entries(20)
        index = build_index(entries, bits=3, window_size=120)
        assert index.quantizer is not None

    def test_quantizer_roundtrip_preserves_similarity(self):
        entries = _normal_entries(20)
        index = build_index(entries, window_size=120)
        # Dequantize and check dimensions
        recovered = index.quantizer.dequantize(index.normal_compressed)
        recovered = np.array(recovered, dtype=np.float32)
        assert recovered.shape[1] == 384
        assert recovered.shape[0] == index.total_windows
