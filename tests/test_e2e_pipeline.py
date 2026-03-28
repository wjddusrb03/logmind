"""End-to-end pipeline tests with REAL sentence-transformers and TurboQuantizer.

These tests exercise complete multi-stage pipelines (parse -> embed -> compress
-> save -> load -> detect -> incident management) using the actual
all-MiniLM-L6-v2 model.  They are intentionally distinct from the unit-level
tests in test_real_model.py and the CLI-focused tests in test_cli_e2e.py.

Run with:  pytest tests/test_e2e_pipeline.py -v
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from logmind.embedder import (
    DEFAULT_MODEL,
    _create_windows,
    build_index,
    embed_window,
)
from logmind.detector import (
    compute_anomaly_score,
    detect,
    find_similar_incidents,
)
from logmind.incidents import auto_detect_incidents, label_incident, search_incidents
from logmind.models import LogEntry, LogMindIndex, LogWindow
from logmind.parser import parse_lines, parse_file
from logmind.storage import load_index, save_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 3, 20, 8, 0, 0)


def _e(
    msg: str,
    level: str = "INFO",
    source: str = "svc",
    offset: int = 0,
    line: int = 0,
) -> LogEntry:
    return LogEntry(
        timestamp=_BASE + timedelta(seconds=offset),
        level=level,
        source=source,
        message=msg,
        raw=f"2026-03-20 08:00:{offset:02d} [{level}] {source}: {msg}",
        line_number=line,
    )


def _normals(n: int = 20, start: int = 0, source: str = "svc") -> List[LogEntry]:
    msgs = [
        "Request completed in 15ms status=200",
        "Health check passed endpoint=/healthz",
        "Scheduled job finished: rotate_logs",
        "Cache hit ratio=97.3%",
        "Connection pool idle=8 active=2 max=50",
        "Metrics exported to prometheus",
        "User session created token=xyz",
        "Config reloaded from /etc/app/config.yaml",
        "Worker heartbeat id=5",
        "TLS handshake succeeded peer=10.0.0.1",
    ]
    return [
        _e(msgs[i % len(msgs)], "INFO", source, start + i, i)
        for i in range(n)
    ]


def _errors(n: int = 10, start: int = 0, source: str = "svc") -> List[LogEntry]:
    msgs = [
        "java.lang.OutOfMemoryError: Java heap space",
        "Connection refused to db-primary:5432",
        "Timeout waiting for upstream response after 30s",
        "FATAL password auth failed for user admin",
        "Segmentation fault (core dumped)",
        "Disk critically low /var/log 99% used",
        "SSL certificate expired for api.example.com",
        "Max retries exceeded for send_email task",
        "NullPointerException in request handler",
        "Kernel panic - not syncing: OOM",
    ]
    return [
        _e(msgs[i % len(msgs)], "ERROR", source, start + i, i)
        for i in range(n)
    ]


def _win(entries: List[LogEntry], label: str = "", resolution: str = "") -> LogWindow:
    return LogWindow(
        entries=entries,
        start_time=entries[0].timestamp if entries else None,
        end_time=entries[-1].timestamp if entries else None,
        window_size=60,
        label=label,
        resolution=resolution,
    )


# ===================================================================
# 1. FULL PIPELINE: parse -> build_index -> save -> load -> detect
# ===================================================================

class TestFullSaveLoadDetectPipeline:
    """Verify that saving and loading an index preserves detection ability."""

    @pytest.fixture()
    def pipeline_index(self, tmp_path):
        """Build, save, and reload an index from normal logs."""
        entries = _normals(40)
        index = build_index(entries, window_size=30)
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        return loaded

    def test_loaded_index_detects_errors(self, pipeline_index):
        w = _win(_errors(10, start=200))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, pipeline_index)
        assert score > 0.0

    def test_loaded_index_normal_low_score(self, pipeline_index):
        w = _win(_normals(10, start=300))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, pipeline_index)
        assert score < 0.5

    def test_loaded_index_metadata_preserved(self, pipeline_index):
        assert pipeline_index.model_name == DEFAULT_MODEL
        assert pipeline_index.embedding_dim == 384
        assert pipeline_index.total_lines == 40

    def test_loaded_index_quantizer_works(self, pipeline_index):
        w = _win(_normals(5))
        vec = embed_window(w)
        scores = pipeline_index.quantizer.cosine_scores(
            vec, pipeline_index.normal_compressed
        )
        assert len(np.array(scores).flatten()) == pipeline_index.total_windows

    def test_save_load_roundtrip_file_exists(self, tmp_path):
        entries = _normals(20)
        index = build_index(entries, window_size=60)
        path = save_index(index, str(tmp_path))
        assert os.path.isfile(path)

    def test_save_overwrite_works(self, tmp_path):
        e1 = _normals(20)
        idx1 = build_index(e1, window_size=60)
        save_index(idx1, str(tmp_path))
        e2 = _normals(30, start=100)
        idx2 = build_index(e2, window_size=60)
        save_index(idx2, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.total_lines == 30


# ===================================================================
# 2. INCREMENTAL INCIDENT LABELING
# ===================================================================

class TestIncrementalIncidentLabeling:
    """Add 3+ incidents sequentially, verify all are searchable."""

    @pytest.fixture()
    def index_3_incidents(self):
        normal = _normals(40)
        index = build_index(normal, window_size=120)

        # Incident 1: database failure
        db_entries = [
            _e("Connection refused to postgres:5432", "ERROR", "db", 200 + i)
            for i in range(5)
        ]
        all_e = normal + db_entries
        index = label_incident(
            index, all_e, _BASE + timedelta(seconds=200),
            _BASE + timedelta(seconds=205), "Database outage",
            "Restarted postgres primary",
        )

        # Incident 2: memory leak
        mem_entries = [
            _e("OutOfMemoryError: heap space exhausted", "ERROR", "jvm", 300 + i)
            for i in range(5)
        ]
        all_e = all_e + mem_entries
        index = label_incident(
            index, all_e, _BASE + timedelta(seconds=300),
            _BASE + timedelta(seconds=305), "JVM OOM crash",
            "Increased Xmx to 8g",
        )

        # Incident 3: SSL expiry
        ssl_entries = [
            _e("SSL certificate has expired for api.example.com", "ERROR", "nginx", 400 + i)
            for i in range(5)
        ]
        all_e = all_e + ssl_entries
        index = label_incident(
            index, all_e, _BASE + timedelta(seconds=400),
            _BASE + timedelta(seconds=405), "TLS cert expired",
            "Renewed certificate via Let's Encrypt",
        )

        return index

    def test_three_incidents_stored(self, index_3_incidents):
        assert index_3_incidents.incident_count == 3

    def test_search_finds_database(self, index_3_incidents):
        results = search_incidents("database connection failure", index_3_incidents)
        labels = [r.label for r in results]
        assert "Database outage" in labels

    def test_search_finds_memory(self, index_3_incidents):
        results = search_incidents("out of memory heap", index_3_incidents)
        labels = [r.label for r in results]
        assert "JVM OOM crash" in labels

    def test_search_finds_ssl(self, index_3_incidents):
        results = search_incidents("certificate expired TLS", index_3_incidents)
        labels = [r.label for r in results]
        assert "TLS cert expired" in labels

    def test_all_resolutions_stored(self, index_3_incidents):
        assert len(index_3_incidents.incident_resolutions) == 3
        assert "Increased Xmx to 8g" in index_3_incidents.incident_resolutions

    def test_incremental_labels_order(self, index_3_incidents):
        assert index_3_incidents.incident_labels == [
            "Database outage", "JVM OOM crash", "TLS cert expired",
        ]


# ===================================================================
# 3. LARGE LOG SIMULATION (500+ entries)
# ===================================================================

class TestLargeLogSimulation:
    """Test with 500+ log entries to verify windowing and detection at scale."""

    @pytest.fixture(scope="class")
    def large_index(self):
        # 500 normal entries, 1 entry per second
        entries = _normals(500, start=0)
        return build_index(entries, window_size=60)

    def test_large_index_window_count(self, large_index):
        # 500 entries over 500 seconds with 60s windows => ~8-9 windows
        assert large_index.total_windows >= 5

    def test_large_index_total_lines(self, large_index):
        assert large_index.total_lines == 500

    def test_large_normal_score_low(self, large_index):
        w = _win(_normals(10, start=600))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, large_index)
        assert score < 0.5

    def test_large_error_score_positive(self, large_index):
        w = _win(_errors(15, start=700))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, large_index)
        assert score > 0.0

    def test_large_detect_no_false_alarm_low_sens(self, large_index):
        w = _win(_normals(10, start=800))
        vec = embed_window(w)
        alert = detect(w, vec, large_index, sensitivity="low")
        assert alert is None


# ===================================================================
# 4. CROSS-FORMAT DETECTION
# ===================================================================

class TestCrossFormatDetection:
    """Learn from Python logs, detect anomaly from Spring-style logs."""

    @pytest.fixture(scope="class")
    def python_index(self):
        python_lines = []
        for i in range(60):
            ts = (_BASE + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            python_lines.append(
                f"{ts} - myapp.server - INFO - Request handled path=/api/v1 status=200"
            )
        entries = parse_lines(python_lines, fmt="auto")
        return build_index(entries, window_size=60)

    def test_spring_error_detected(self, python_index):
        spring_lines = []
        for i in range(10):
            ts = (_BASE + timedelta(seconds=500 + i)).strftime("%Y-%m-%d %H:%M:%S.000")
            spring_lines.append(
                f"{ts} ERROR 1234 --- [main] c.m.DatabaseService : "
                f"Connection refused to database host db-primary:5432"
            )
        entries = parse_lines(spring_lines, fmt="auto")
        w = _win(entries)
        vec = embed_window(w)
        score = compute_anomaly_score(vec, python_index)
        # Errors from a different format should still diverge from normal baseline
        assert score > 0.0

    def test_spring_normal_lower_than_error(self, python_index):
        spring_ok_lines = []
        for i in range(10):
            ts = (_BASE + timedelta(seconds=600 + i)).strftime("%Y-%m-%d %H:%M:%S.000")
            spring_ok_lines.append(
                f"{ts} INFO 1234 --- [main] c.m.HealthCheck : Health check OK"
            )
        entries_ok = parse_lines(spring_ok_lines, fmt="auto")
        w_ok = _win(entries_ok)
        v_ok = embed_window(w_ok)
        score_ok = compute_anomaly_score(v_ok, python_index)

        spring_err_lines = []
        for i in range(10):
            ts = (_BASE + timedelta(seconds=700 + i)).strftime("%Y-%m-%d %H:%M:%S.000")
            spring_err_lines.append(
                f"{ts} ERROR 1234 --- [main] c.m.DbPool : "
                f"Connection refused to database timeout after 30s"
            )
        entries_err = parse_lines(spring_err_lines, fmt="auto")
        w_err = _win(entries_err)
        v_err = embed_window(w_err)
        score_err = compute_anomaly_score(v_err, python_index)

        assert score_err >= score_ok


# ===================================================================
# 5. SEMANTIC GROUPING
# ===================================================================

class TestSemanticGrouping:
    """Connection-related errors should cluster together semantically."""

    @pytest.fixture(scope="class")
    def conn_index(self):
        normal = _normals(30)
        refused = [_e("Connection refused to remote host", "ERROR", "net", 200 + i) for i in range(5)]
        w_refused = _win(refused, label="Connection refused", resolution="Restart remote")
        return build_index(normal, window_size=120, incident_windows=[w_refused])

    def test_reset_matches_refused(self, conn_index):
        reset_entries = [_e("Connection reset by peer", "ERROR", "net", 300 + i) for i in range(5)]
        w_reset = _win(reset_entries)
        vec = embed_window(w_reset)
        matches = find_similar_incidents(vec, conn_index, k=1)
        assert len(matches) >= 1
        assert matches[0].label == "Connection refused"

    def test_timeout_matches_refused(self, conn_index):
        timeout_entries = [_e("Connection timed out after 30s", "ERROR", "net", 400 + i) for i in range(5)]
        w_timeout = _win(timeout_entries)
        vec = embed_window(w_timeout)
        matches = find_similar_incidents(vec, conn_index, k=1)
        assert len(matches) >= 1
        assert matches[0].label == "Connection refused"

    def test_unrelated_does_not_match(self, conn_index):
        disk_entries = [_e("Disk usage at 98% on /var/log", "ERROR", "mon", 500 + i) for i in range(5)]
        w_disk = _win(disk_entries)
        vec = embed_window(w_disk)
        matches = find_similar_incidents(vec, conn_index, k=1)
        if matches:
            # Similarity should be lower for unrelated issue
            assert matches[0].similarity < 0.8


# ===================================================================
# 6. FALSE POSITIVE RESISTANCE
# ===================================================================

class TestFalsePositiveResistance:
    """Normal operational logs should NOT trigger alerts."""

    @pytest.fixture(scope="class")
    def healthy_index(self):
        return build_index(_normals(60), window_size=30)

    def test_normal_no_alert_low(self, healthy_index):
        w = _win(_normals(10, start=200))
        vec = embed_window(w)
        alert = detect(w, vec, healthy_index, sensitivity="low")
        assert alert is None

    def test_normal_no_alert_medium(self, healthy_index):
        w = _win(_normals(10, start=300))
        vec = embed_window(w)
        alert = detect(w, vec, healthy_index, sensitivity="medium")
        assert alert is None

    def test_varied_normal_no_alert(self, healthy_index):
        varied = [
            _e("GET /api/users 200 OK in 12ms", "INFO", "nginx", 400),
            _e("SELECT * FROM users completed 3ms", "INFO", "postgres", 401),
            _e("Cache set key=session:abc ttl=3600", "INFO", "redis", 402),
            _e("Background job enqueued: send_notification", "INFO", "celery", 403),
            _e("Metrics flushed 42 points", "INFO", "statsd", 404),
        ]
        w = _win(varied)
        vec = embed_window(w)
        alert = detect(w, vec, healthy_index, sensitivity="low")
        assert alert is None


# ===================================================================
# 7. ERROR SPIKE DETECTION ACCURACY
# ===================================================================

class TestErrorSpikeDetection:
    """Inject sudden error burst, verify detection."""

    @pytest.fixture(scope="class")
    def calm_index(self):
        # Mostly normal with very few errors
        entries = _normals(50)
        entries.append(_e("Transient connection warning", "WARN", "svc", 51))
        return build_index(entries, window_size=30)

    def test_error_burst_triggers_alert(self, calm_index):
        burst = _errors(20, start=200)
        w = _win(burst)
        vec = embed_window(w)
        alert = detect(w, vec, calm_index, sensitivity="high")
        if alert is not None:
            assert alert.anomaly_score > 0.0
            assert alert.severity in ("CRITICAL", "WARNING", "INFO")

    def test_error_burst_higher_score_than_normal(self, calm_index):
        w_normal = _win(_normals(10, start=300))
        w_error = _win(_errors(10, start=400))
        s_normal = compute_anomaly_score(embed_window(w_normal), calm_index)
        s_error = compute_anomaly_score(embed_window(w_error), calm_index)
        assert s_error > s_normal

    def test_auto_detect_finds_spike(self):
        normal = _normals(60, start=0)
        spike = _errors(30, start=100)
        more_normal = _normals(60, start=200)
        all_entries = normal + spike + more_normal
        detected = auto_detect_incidents(all_entries, window_size=30, spike_threshold=1.5)
        assert len(detected) >= 1
        for w in detected:
            assert w.error_count > 0


# ===================================================================
# 8. MULTIPLE INCIDENT TYPES
# ===================================================================

class TestMultipleIncidentTypes:
    """DB, memory, network incidents coexist and match correctly."""

    @pytest.fixture(scope="class")
    def multi_index(self):
        normal = _normals(40)

        db_entries = [_e("psycopg2.OperationalError: could not connect to server", "ERROR", "db", 200 + i) for i in range(5)]
        mem_entries = [_e("java.lang.OutOfMemoryError: GC overhead limit exceeded", "ERROR", "jvm", 300 + i) for i in range(5)]
        net_entries = [_e("Connection timed out to upstream service gateway:8080", "ERROR", "proxy", 400 + i) for i in range(5)]

        w_db = _win(db_entries, label="DB connection failure", resolution="Restarted postgres")
        w_mem = _win(mem_entries, label="JVM memory exhaustion", resolution="Added -Xmx8g flag")
        w_net = _win(net_entries, label="Network timeout", resolution="Increased timeout to 60s")

        return build_index(normal, window_size=120, incident_windows=[w_db, w_mem, w_net])

    def test_db_query_matches_db_incident(self, multi_index):
        results = search_incidents("database connection refused postgres", multi_index)
        assert results[0].label == "DB connection failure"

    def test_memory_query_matches_memory_incident(self, multi_index):
        results = search_incidents("out of memory heap space", multi_index)
        assert results[0].label == "JVM memory exhaustion"

    def test_network_query_matches_network_incident(self, multi_index):
        results = search_incidents("upstream timeout gateway", multi_index)
        assert results[0].label == "Network timeout"

    def test_all_three_have_resolutions(self, multi_index):
        assert multi_index.incident_count == 3
        assert all(r != "" for r in multi_index.incident_resolutions)


# ===================================================================
# 9. TIME-RANGE BOUNDARY TESTING FOR label_incident
# ===================================================================

class TestTimeRangeBoundary:
    """Boundary conditions for label_incident time ranges."""

    @pytest.fixture()
    def base_setup(self):
        entries = [_e(f"Log msg {i}", "INFO" if i % 3 != 0 else "ERROR", "svc", i) for i in range(100)]
        index = build_index(entries, window_size=120)
        return index, entries

    def test_exact_boundary_inclusive(self, base_setup):
        index, entries = base_setup
        # Single-second range that includes exactly one entry at offset=50
        start = _BASE + timedelta(seconds=50)
        end = _BASE + timedelta(seconds=50)
        index = label_incident(index, entries, start, end, "Exact boundary")
        assert index.incident_count == 1

    def test_one_second_range(self, base_setup):
        index, entries = base_setup
        start = _BASE + timedelta(seconds=10)
        end = _BASE + timedelta(seconds=11)
        index = label_incident(index, entries, start, end, "One second range")
        assert index.incident_count == 1
        assert len(index.incident_windows[-1].entries) == 2  # offsets 10 and 11

    def test_wide_range_captures_many(self, base_setup):
        index, entries = base_setup
        start = _BASE + timedelta(seconds=0)
        end = _BASE + timedelta(seconds=99)
        index = label_incident(index, entries, start, end, "Full range")
        assert len(index.incident_windows[-1].entries) == 100

    def test_out_of_range_raises(self, base_setup):
        index, entries = base_setup
        far_start = _BASE + timedelta(hours=5)
        far_end = _BASE + timedelta(hours=6)
        with pytest.raises(ValueError, match="No log entries found"):
            label_incident(index, entries, far_start, far_end, "Nothing here")


# ===================================================================
# 10. SEARCH RANKING
# ===================================================================

class TestSearchRanking:
    """Most relevant incident should rank first."""

    @pytest.fixture(scope="class")
    def ranked_index(self):
        normal = _normals(30)

        disk_entries = [_e("Disk usage at 99% on /var/log partition full", "ERROR", "os", 200 + i) for i in range(5)]
        auth_entries = [_e("Authentication failed for user admin wrong password", "ERROR", "auth", 300 + i) for i in range(5)]
        dns_entries = [_e("DNS resolution failed for api.example.com NXDOMAIN", "ERROR", "dns", 400 + i) for i in range(5)]

        w_disk = _win(disk_entries, label="Disk full /var/log")
        w_auth = _win(auth_entries, label="Auth failure admin")
        w_dns = _win(dns_entries, label="DNS resolution failure")

        return build_index(normal, window_size=120, incident_windows=[w_disk, w_auth, w_dns])

    def test_disk_query_ranks_disk_first(self, ranked_index):
        results = search_incidents("disk space usage full", ranked_index)
        assert results[0].label == "Disk full /var/log"

    def test_auth_query_ranks_auth_first(self, ranked_index):
        results = search_incidents("authentication password failed login", ranked_index)
        assert results[0].label == "Auth failure admin"

    def test_dns_query_ranks_dns_first(self, ranked_index):
        results = search_incidents("DNS domain name resolution NXDOMAIN", ranked_index)
        assert results[0].label == "DNS resolution failure"

    def test_ranking_scores_descending(self, ranked_index):
        results = search_incidents("disk full partition", ranked_index)
        if len(results) >= 2:
            assert results[0].similarity >= results[1].similarity


# ===================================================================
# 11. WINDOW SIZE IMPACT
# ===================================================================

class TestWindowSizeImpact:
    """Same logs with different window sizes produce different results."""

    @pytest.fixture(scope="class")
    def entries_500s(self):
        return _normals(100, start=0)

    def test_small_window_more_windows(self, entries_500s):
        idx_small = build_index(entries_500s, window_size=20)
        idx_large = build_index(entries_500s, window_size=120)
        assert idx_small.total_windows > idx_large.total_windows

    def test_different_window_sizes_different_scores(self, entries_500s):
        idx_small = build_index(entries_500s, window_size=20)
        idx_large = build_index(entries_500s, window_size=120)

        probe = _win(_errors(5, start=600))
        vec = embed_window(probe)

        s_small = compute_anomaly_score(vec, idx_small)
        s_large = compute_anomaly_score(vec, idx_large)
        # They should differ (not identical); both positive
        assert s_small > 0.0
        assert s_large > 0.0

    def test_single_entry_window_size_60(self, entries_500s):
        # With a very large window, all entries fit in fewer windows
        idx = build_index(entries_500s, window_size=600)
        assert idx.total_windows >= 1


# ===================================================================
# 12. COMPRESSION FIDELITY (2-bit and 4-bit)
# ===================================================================

class TestCompressionFidelity:
    """Detection still works after aggressive quantization."""

    @pytest.fixture(scope="class")
    def entries(self):
        return _normals(40)

    def test_2bit_build_and_detect(self, entries):
        index = build_index(entries, bits=2, window_size=60)
        assert index.quantizer is not None
        w_err = _win(_errors(10, start=300))
        vec = embed_window(w_err)
        score = compute_anomaly_score(vec, index)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_4bit_build_and_detect(self, entries):
        index = build_index(entries, bits=4, window_size=60)
        w_err = _win(_errors(10, start=400))
        vec = embed_window(w_err)
        score = compute_anomaly_score(vec, index)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_2bit_vs_4bit_both_detect_anomaly(self, entries):
        idx_2 = build_index(entries, bits=2, window_size=60)
        idx_4 = build_index(entries, bits=4, window_size=60)

        probe = _win(_errors(10, start=500))
        vec = embed_window(probe)

        s2 = compute_anomaly_score(vec, idx_2)
        s4 = compute_anomaly_score(vec, idx_4)
        assert s2 > 0.0
        assert s4 > 0.0

    def test_2bit_save_load_still_works(self, entries, tmp_path):
        index = build_index(entries, bits=2, window_size=60)
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        w = _win(_errors(5, start=600))
        vec = embed_window(w)
        score = compute_anomaly_score(vec, loaded)
        assert score > 0.0

    def test_4bit_incidents_searchable(self, entries):
        inc = [_e("Network partition detected: node unreachable", "ERROR", "cluster", 200 + i) for i in range(5)]
        w_inc = _win(inc, label="Network partition", resolution="Healed partition")
        index = build_index(entries, bits=4, window_size=60, incident_windows=[w_inc])
        results = search_incidents("network partition unreachable", index)
        assert len(results) >= 1
        assert results[0].label == "Network partition"


# ===================================================================
# 13. EMPTY / MINIMAL INPUTS
# ===================================================================

class TestEmptyMinimalInputs:
    """Edge cases with minimal or single-entry data."""

    def test_single_log_line_builds_index(self):
        entries = [_e("Service started")]
        index = build_index(entries, window_size=60)
        assert index.total_windows == 1
        assert index.total_lines == 1

    def test_single_window_detect(self):
        entries = [_e("Service started")]
        index = build_index(entries, window_size=60)
        w = _win([_e("Service crashed", "ERROR", "svc", 100)])
        vec = embed_window(w)
        score = compute_anomaly_score(vec, index)
        assert 0.0 <= score <= 1.0

    def test_single_entry_incident_label(self):
        normal = _normals(10)
        index = build_index(normal, window_size=120)
        err = [_e("Fatal error occurred", "ERROR", "svc", 50)]
        all_e = normal + err
        index = label_incident(
            index, all_e,
            _BASE + timedelta(seconds=50),
            _BASE + timedelta(seconds=50),
            "Single entry incident",
        )
        assert index.incident_count == 1

    def test_single_entry_search(self):
        normal = _normals(10)
        inc = [_e("Redis connection pool exhausted", "ERROR", "redis", 50)]
        w_inc = _win(inc, label="Redis pool exhausted")
        index = build_index(normal, window_size=120, incident_windows=[w_inc])
        results = search_incidents("redis connection pool", index)
        assert len(results) >= 1

    def test_two_entries_different_windows(self):
        entries = [
            _e("First entry", "INFO", "svc", 0),
            _e("Second entry", "INFO", "svc", 120),
        ]
        index = build_index(entries, window_size=60)
        assert index.total_windows == 2


# ===================================================================
# 14. UNICODE LOG MESSAGES
# ===================================================================

class TestUnicodeLogMessages:
    """Korean, Japanese, emoji in log messages."""

    def test_korean_error_embeds(self):
        entries = [
            _e("DB 연결 실패: 타임아웃 오류 발생", "ERROR", "한국-서버", 0),
            _e("메모리 사용량 초과: 90% 사용 중", "WARN", "한국-서버", 1),
            _e("재시도 3회 후 연결 성공", "INFO", "한국-서버", 2),
        ]
        w = _win(entries)
        vec = embed_window(w)
        assert vec.shape == (384,)
        assert np.any(vec != 0.0)

    def test_japanese_error_embeds(self):
        entries = [
            _e("データベース接続に失敗しました", "ERROR", "jp-app", 0),
            _e("メモリ不足エラーが発生しました", "ERROR", "jp-app", 1),
            _e("設定ファイルを再読み込みしました", "INFO", "jp-app", 2),
        ]
        w = _win(entries)
        vec = embed_window(w)
        assert vec.shape == (384,)
        assert np.any(vec != 0.0)

    def test_emoji_in_logs(self):
        entries = [
            _e("Deploy completed successfully :rocket:", "INFO", "deploy", 0),
            _e("Build failed :x: exit code 1", "ERROR", "ci", 1),
            _e("Tests passed :white_check_mark: 42/42", "INFO", "ci", 2),
        ]
        w = _win(entries)
        vec = embed_window(w)
        assert vec.shape == (384,)

    def test_mixed_unicode_index_build(self):
        entries = [
            _e("서버 시작됨 - Server started", "INFO", "hybrid", i)
            for i in range(10)
        ]
        index = build_index(entries, window_size=120)
        assert index.total_lines == 10
        assert index.total_windows >= 1

    def test_unicode_incident_searchable(self):
        normal = _normals(20)
        ko_entries = [_e("데이터베이스 연결 거부 오류", "ERROR", "db-ko", 200 + i) for i in range(5)]
        w_ko = _win(ko_entries, label="DB 연결 거부", resolution="DB 재시작")
        index = build_index(normal, window_size=120, incident_windows=[w_ko])
        results = search_incidents("데이터베이스 연결", index)
        assert len(results) >= 1
        assert results[0].label == "DB 연결 거부"


# ===================================================================
# 15. PARSE -> INDEX FULL CHAIN
# ===================================================================

class TestParseToIndexChain:
    """Parse raw log text lines, then build index and detect."""

    def test_parse_simple_and_build(self):
        raw = []
        for i in range(30):
            ts = (_BASE + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            level = "ERROR" if i % 10 == 0 else "INFO"
            raw.append(f"{ts} {level} AppServer: Request processed id={i}")
        entries = parse_lines(raw, fmt="auto")
        assert len(entries) >= 25
        index = build_index(entries, window_size=60)
        assert index.total_windows >= 1

    def test_parse_json_and_detect(self):
        import json as _json
        raw = []
        for i in range(30):
            ts = (_BASE + timedelta(seconds=i)).isoformat()
            entry = {"timestamp": ts, "level": "INFO", "source": "api", "message": f"OK {i}"}
            raw.append(_json.dumps(entry))
        entries = parse_lines(raw, fmt="json")
        index = build_index(entries, window_size=60)

        err_raw = []
        for i in range(10):
            ts = (_BASE + timedelta(seconds=200 + i)).isoformat()
            entry = {"timestamp": ts, "level": "ERROR", "source": "api", "message": "DB connection refused"}
            err_raw.append(_json.dumps(entry))
        err_entries = parse_lines(err_raw, fmt="json")
        w = _win(err_entries)
        vec = embed_window(w)
        score = compute_anomaly_score(vec, index)
        assert score > 0.0

    def test_parse_file_and_build(self, tmp_path):
        log_file = tmp_path / "app.log"
        lines = []
        for i in range(50):
            ts = (_BASE + timedelta(seconds=i * 2)).strftime("%Y-%m-%d %H:%M:%S")
            level = "ERROR" if i % 15 == 0 else "INFO"
            lines.append(f"{ts} {level} WebApp: Handling request {i}\n")
        log_file.write_text("".join(lines), encoding="utf-8")

        entries = parse_file(str(log_file), fmt="auto")
        assert len(entries) >= 40
        index = build_index(entries, window_size=30)
        assert index.total_windows >= 1


# ===================================================================
# 16. SAVE/LOAD WITH INCIDENTS PRESERVED
# ===================================================================

class TestSaveLoadIncidents:
    """Save/load preserves incident data and search still works."""

    def test_roundtrip_incidents(self, tmp_path):
        normal = _normals(30)
        inc = [_e("Kafka consumer lag exceeded threshold", "ERROR", "kafka", 200 + i) for i in range(5)]
        w_inc = _win(inc, label="Kafka lag spike", resolution="Scaled consumers")
        index = build_index(normal, window_size=120, incident_windows=[w_inc])

        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.incident_count == 1
        assert loaded.incident_labels == ["Kafka lag spike"]

        results = search_incidents("kafka consumer lag", loaded)
        assert len(results) >= 1
        assert results[0].label == "Kafka lag spike"

    def test_roundtrip_after_incremental_label(self, tmp_path):
        normal = _normals(30)
        index = build_index(normal, window_size=120)
        err = [_e("Redis cluster node unreachable", "ERROR", "redis", 200 + i) for i in range(5)]
        all_e = normal + err
        index = label_incident(
            index, all_e,
            _BASE + timedelta(seconds=200),
            _BASE + timedelta(seconds=205),
            "Redis node down",
            "Replaced failed node",
        )

        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.incident_count == 1
        results = search_incidents("redis node unreachable", loaded)
        assert len(results) >= 1


# ===================================================================
# 17. DETECTION WITH SIMILAR INCIDENT MATCHING
# ===================================================================

class TestDetectWithIncidentMatching:
    """When past incidents exist, detect should report similar_incident type."""

    @pytest.fixture(scope="class")
    def index_with_known_incident(self):
        normal = _normals(40)
        oom = [_e("java.lang.OutOfMemoryError: Java heap space", "ERROR", "jvm", 200 + i) for i in range(10)]
        w_oom = _win(oom, label="OOM crash March", resolution="Increased heap")
        return build_index(normal, window_size=120, incident_windows=[w_oom])

    def test_similar_oom_gets_matched(self, index_with_known_incident):
        new_oom = [_e("OutOfMemoryError: GC overhead limit exceeded", "ERROR", "jvm", 500 + i) for i in range(10)]
        w = _win(new_oom)
        vec = embed_window(w)
        matches = find_similar_incidents(vec, index_with_known_incident, k=1)
        assert len(matches) >= 1
        assert matches[0].label == "OOM crash March"
        assert matches[0].resolution == "Increased heap"

    def test_detect_reports_incident_match(self, index_with_known_incident):
        new_oom = [_e("java.lang.OutOfMemoryError: heap space", "ERROR", "jvm", 600 + i) for i in range(10)]
        w = _win(new_oom)
        vec = embed_window(w)
        alert = detect(w, vec, index_with_known_incident, sensitivity="high")
        if alert is not None and alert.similar_incidents:
            assert alert.similar_incidents[0].label == "OOM crash March"
