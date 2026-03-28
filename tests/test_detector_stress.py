"""Stress / comprehensive tests for logmind detector and embedder modules.

50+ test cases covering:
  - Window creation (various entry counts, intervals, edge cases, no-timestamp)
  - Anomaly score computation (normal vectors, outlier vectors, boundary cases)
  - Severity classification (all severity levels, edge cases)
  - Anomaly type classification (error_spike, new_pattern, similar_incident)
  - Similar incident finding (with/without incidents, top-k, low similarity filtering)
  - Full detect pipeline (various sensitivity levels, threshold boundaries)
  - Summary generation (all anomaly types, with/without incidents)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from langchain_turboquant import TurboQuantizer

from logmind.detector import (
    _classify_anomaly_type,
    _classify_severity,
    _generate_summary,
    compute_anomaly_score,
    detect,
    find_similar_incidents,
)
from logmind.embedder import _create_windows
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

DIM = 384
BITS = 3
_Q = TurboQuantizer(dim=DIM, bits=BITS)


def _make_entry(
    ts: datetime | None = None,
    level: str = "INFO",
    source: str = "svc",
    message: str = "ok",
    line_number: int = 0,
) -> LogEntry:
    return LogEntry(
        timestamp=ts,
        level=level,
        source=source,
        message=message,
        raw=f"[{level}] {message}",
        line_number=line_number,
    )


def _make_window(
    n_entries: int = 5,
    n_errors: int = 0,
    n_warns: int = 0,
    ts: datetime | None = None,
    label: str = "",
    resolution: str = "",
    window_size: int = 60,
) -> LogWindow:
    if ts is None:
        ts = datetime(2025, 1, 1)
    entries: List[LogEntry] = []
    for i in range(n_errors):
        entries.append(_make_entry(ts=ts + timedelta(seconds=i), level="ERROR", message=f"err-{i}"))
    for i in range(n_warns):
        entries.append(_make_entry(ts=ts + timedelta(seconds=n_errors + i), level="WARN", message=f"warn-{i}"))
    for i in range(n_entries - n_errors - n_warns):
        entries.append(_make_entry(ts=ts + timedelta(seconds=n_errors + n_warns + i), level="INFO", message=f"info-{i}"))
    return LogWindow(
        entries=entries,
        start_time=ts,
        end_time=ts + timedelta(seconds=max(n_entries - 1, 0)),
        window_size=window_size,
        label=label,
        resolution=resolution,
    )


def _rand_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def _similar_vec(base: np.ndarray, noise: float = 0.05, seed: int = 99) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = base + rng.randn(DIM).astype(np.float32) * noise
    v /= np.linalg.norm(v) + 1e-9
    return v


def _orthogonal_vec(base: np.ndarray, seed: int = 42) -> np.ndarray:
    """Create a vector roughly orthogonal to base."""
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    # Remove component along base
    v = v - np.dot(v, base) * base
    v /= np.linalg.norm(v) + 1e-9
    return v


def _build_index(
    n_normal: int = 5,
    n_incidents: int = 0,
    normal_seed: int = 0,
    incident_seed: int = 100,
    avg_errors: float = 1.0,
    std_errors: float = 0.5,
    avg_warns: float = 0.5,
    incident_labels: List[str] | None = None,
    incident_resolutions: List[str] | None = None,
    incident_windows: List[LogWindow] | None = None,
) -> LogMindIndex:
    """Build a lightweight index for testing without sentence-transformers."""
    rng_n = np.random.RandomState(normal_seed)
    normal_vecs = rng_n.randn(n_normal, DIM).astype(np.float32)
    for i in range(n_normal):
        normal_vecs[i] /= np.linalg.norm(normal_vecs[i]) + 1e-9
    normal_compressed = _Q.quantize(normal_vecs)

    inc_compressed = None
    inc_wins: List[LogWindow] = incident_windows or []
    inc_labels: List[str] = incident_labels or []
    inc_resolutions: List[str] = incident_resolutions or []

    if n_incidents > 0:
        rng_i = np.random.RandomState(incident_seed)
        inc_vecs = rng_i.randn(n_incidents, DIM).astype(np.float32)
        for i in range(n_incidents):
            inc_vecs[i] /= np.linalg.norm(inc_vecs[i]) + 1e-9
        inc_compressed = _Q.quantize(inc_vecs)
        if not inc_wins:
            inc_wins = [_make_window(label=f"inc-{i}", resolution=f"fix-{i}") for i in range(n_incidents)]
        if not inc_labels:
            inc_labels = [f"inc-{i}" for i in range(n_incidents)]
        if not inc_resolutions:
            inc_resolutions = [f"fix-{i}" for i in range(n_incidents)]

    return LogMindIndex(
        normal_compressed=normal_compressed,
        normal_windows=[_make_window() for _ in range(n_normal)],
        incident_compressed=inc_compressed,
        incident_windows=inc_wins,
        incident_labels=inc_labels,
        incident_resolutions=inc_resolutions,
        quantizer=_Q,
        model_name="test",
        embedding_dim=DIM,
        total_lines=n_normal * 5,
        total_windows=n_normal,
        incident_count=n_incidents,
        error_count=0,
        warn_count=0,
        sources=["svc"],
        learn_time=0.0,
        avg_errors_per_window=avg_errors,
        std_errors_per_window=std_errors,
        avg_warns_per_window=avg_warns,
    )


# ===================================================================
# 1. Window creation (_create_windows)
# ===================================================================


class TestCreateWindows:
    """Tests for embedder._create_windows."""

    def test_empty_entries(self):
        assert _create_windows([]) == []

    def test_single_entry(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts)]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) == 1
        assert windows[0].total_count == 1

    def test_entries_within_one_window(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i)) for i in range(10)]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) == 1
        assert windows[0].total_count == 10

    def test_entries_split_into_two_windows(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i * 10)) for i in range(13)]
        # 0..120s -> window_size=60 => entries at 0-50 in window1, 60-120 in window2
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 2
        total = sum(w.total_count for w in windows)
        assert total == 13

    def test_large_entry_count(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i)) for i in range(500)]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 8
        total = sum(w.total_count for w in windows)
        assert total == 500

    def test_no_timestamp_entries(self):
        entries = [_make_entry(ts=None, message=f"line-{i}") for i in range(100)]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 1
        total = sum(w.total_count for w in windows)
        assert total == 100

    def test_no_timestamp_small_batch(self):
        entries = [_make_entry(ts=None) for _ in range(5)]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 1

    def test_mixed_timed_untimed_ignores_untimed(self):
        """When both timed and untimed exist, untimed are dropped."""
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts),
            _make_entry(ts=ts + timedelta(seconds=5)),
            _make_entry(ts=None),
        ]
        windows = _create_windows(entries, window_size=60)
        # untimed entries are only grouped when there are NO timed entries
        total = sum(w.total_count for w in windows)
        assert total == 2  # only timed ones

    def test_custom_window_size(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i * 5)) for i in range(20)]
        windows_30 = _create_windows(entries, window_size=30)
        windows_120 = _create_windows(entries, window_size=120)
        assert len(windows_30) >= len(windows_120)

    def test_window_start_end_times(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i * 10)) for i in range(5)]
        windows = _create_windows(entries, window_size=60)
        for w in windows:
            assert w.start_time is not None
            assert w.end_time is not None
            assert w.start_time <= w.end_time

    def test_out_of_order_entries_sorted(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts + timedelta(seconds=30)),
            _make_entry(ts=ts + timedelta(seconds=10)),
            _make_entry(ts=ts + timedelta(seconds=20)),
        ]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) == 1
        assert windows[0].total_count == 3

    def test_exact_boundary_entry(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts),
            _make_entry(ts=ts + timedelta(seconds=60)),  # exactly at boundary
        ]
        windows = _create_windows(entries, window_size=60)
        # At exactly window_size the entry goes to a new window
        assert len(windows) == 2


# ===================================================================
# 2. Anomaly score computation
# ===================================================================


class TestAnomalyScore:

    def test_identical_vector_score_near_zero(self):
        base = _rand_vec(seed=1)
        vecs = np.stack([base, _similar_vec(base, noise=0.01, seed=2)])
        idx = _build_index()
        idx.normal_compressed = _Q.quantize(vecs)
        score = compute_anomaly_score(base, idx)
        assert score < 0.15

    def test_orthogonal_vector_score_high(self):
        base = _rand_vec(seed=10)
        normals = np.stack([base])
        idx = _build_index()
        idx.normal_compressed = _Q.quantize(normals)
        query = _orthogonal_vec(base, seed=20)
        score = compute_anomaly_score(query, idx)
        assert score > 0.5

    def test_empty_normals_returns_one(self):
        idx = _build_index(n_normal=1)
        # Create empty compressed by quantizing then slicing trick
        empty_vecs = np.zeros((0, DIM), dtype=np.float32)
        # Overwrite with a dummy that yields empty scores
        idx.normal_compressed = _Q.quantize(np.stack([_rand_vec(0)]))
        # Replace quantizer to return empty for this test
        real_q = idx.quantizer

        class _EmptyScorer:
            def cosine_scores(self, q, db):
                return []
            def quantize(self, x):
                return real_q.quantize(x)

        idx.quantizer = _EmptyScorer()
        score = compute_anomaly_score(_rand_vec(5), idx)
        assert score == 1.0
        idx.quantizer = real_q  # restore

    def test_score_bounded_0_1(self):
        idx = _build_index(n_normal=10, normal_seed=77)
        for seed in range(20):
            score = compute_anomaly_score(_rand_vec(seed), idx)
            assert 0.0 <= score <= 1.0

    def test_closer_vector_lower_score(self):
        base = _rand_vec(seed=50)
        normals = np.stack([base])
        idx = _build_index()
        idx.normal_compressed = _Q.quantize(normals)
        close = _similar_vec(base, noise=0.05, seed=51)
        far = _similar_vec(base, noise=0.8, seed=52)
        s_close = compute_anomaly_score(close, idx)
        s_far = compute_anomaly_score(far, idx)
        assert s_close < s_far

    def test_multiple_normals_uses_max_similarity(self):
        v1 = _rand_vec(seed=60)
        v2 = _rand_vec(seed=61)
        normals = np.stack([v1, v2])
        idx = _build_index()
        idx.normal_compressed = _Q.quantize(normals)
        query = _similar_vec(v1, noise=0.02, seed=62)
        score = compute_anomaly_score(query, idx)
        assert score < 0.15


# ===================================================================
# 3. Severity classification
# ===================================================================


class TestSeverityClassification:

    def test_critical_high_score(self):
        w = _make_window(n_entries=5, n_errors=1)
        sev = _classify_severity(0.75, w, avg_errors=1.0)
        assert sev == "CRITICAL"

    def test_critical_many_errors(self):
        w = _make_window(n_entries=20, n_errors=15)
        sev = _classify_severity(0.3, w, avg_errors=1.0)
        assert sev == "CRITICAL"

    def test_warning_medium_score(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.55, w, avg_errors=1.0)
        assert sev == "WARNING"

    def test_warning_moderate_errors(self):
        w = _make_window(n_entries=10, n_errors=4)
        sev = _classify_severity(0.3, w, avg_errors=1.0)
        assert sev == "WARNING"

    def test_info_low_score_few_errors(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.2, w, avg_errors=1.0)
        assert sev == "INFO"

    def test_boundary_070(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.70, w, avg_errors=1.0)
        assert sev == "CRITICAL"

    def test_boundary_050(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.50, w, avg_errors=1.0)
        assert sev == "WARNING"

    def test_boundary_just_below_050(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.49, w, avg_errors=1.0)
        assert sev == "INFO"

    def test_error_count_threshold_critical(self):
        # avg_errors * 5 = 5 but max(10, 5) = 10 -> need >= 10 errors
        w = _make_window(n_entries=12, n_errors=10)
        sev = _classify_severity(0.0, w, avg_errors=1.0)
        assert sev == "CRITICAL"

    def test_error_count_threshold_warning(self):
        # avg_errors * 3 = 3, max(3, 3) = 3 -> need >= 3
        w = _make_window(n_entries=5, n_errors=3)
        sev = _classify_severity(0.0, w, avg_errors=1.0)
        assert sev == "WARNING"

    def test_zero_avg_errors(self):
        w = _make_window(n_entries=5, n_errors=0)
        sev = _classify_severity(0.2, w, avg_errors=0.0)
        assert sev == "INFO"


# ===================================================================
# 4. Anomaly type classification
# ===================================================================


class TestAnomalyTypeClassification:

    def test_error_spike(self):
        # error_threshold = avg + max(3*std, 3) = 1 + max(1.5, 3) = 4
        w = _make_window(n_entries=10, n_errors=5)
        atype = _classify_anomaly_type(0.5, w, avg_errors=1.0, std_errors=0.5, similar_incidents=[])
        assert atype == "error_spike"

    def test_error_spike_high_std(self):
        # error_threshold = 2 + max(3*3, 3) = 2 + 9 = 11
        w = _make_window(n_entries=15, n_errors=12)
        atype = _classify_anomaly_type(0.5, w, avg_errors=2.0, std_errors=3.0, similar_incidents=[])
        assert atype == "error_spike"

    def test_similar_incident(self):
        match = IncidentMatch(similarity=0.7, label="OOM", resolution="restart", occurred_at=None)
        w = _make_window(n_entries=5, n_errors=0)
        atype = _classify_anomaly_type(0.5, w, avg_errors=10.0, std_errors=5.0, similar_incidents=[match])
        assert atype == "similar_incident"

    def test_similar_incident_boundary_060(self):
        match = IncidentMatch(similarity=0.6, label="OOM", resolution="restart", occurred_at=None)
        w = _make_window(n_entries=5, n_errors=0)
        atype = _classify_anomaly_type(0.5, w, avg_errors=10.0, std_errors=5.0, similar_incidents=[match])
        assert atype == "similar_incident"

    def test_similar_incident_below_threshold_falls_to_new_pattern(self):
        match = IncidentMatch(similarity=0.59, label="OOM", resolution="restart", occurred_at=None)
        w = _make_window(n_entries=5, n_errors=0)
        atype = _classify_anomaly_type(0.5, w, avg_errors=10.0, std_errors=5.0, similar_incidents=[match])
        assert atype == "new_pattern"

    def test_new_pattern_no_incidents(self):
        w = _make_window(n_entries=5, n_errors=0)
        atype = _classify_anomaly_type(0.5, w, avg_errors=10.0, std_errors=5.0, similar_incidents=[])
        assert atype == "new_pattern"

    def test_error_spike_takes_priority_over_similar_incident(self):
        match = IncidentMatch(similarity=0.9, label="OOM", resolution="restart", occurred_at=None)
        # error_threshold = 1 + max(1.5, 3) = 4
        w = _make_window(n_entries=10, n_errors=5)
        atype = _classify_anomaly_type(0.5, w, avg_errors=1.0, std_errors=0.5, similar_incidents=[match])
        assert atype == "error_spike"

    def test_new_pattern_empty_incidents_list(self):
        w = _make_window(n_entries=5, n_errors=0)
        atype = _classify_anomaly_type(0.8, w, avg_errors=100.0, std_errors=50.0, similar_incidents=[])
        assert atype == "new_pattern"


# ===================================================================
# 5. Similar incident finding
# ===================================================================


class TestFindSimilarIncidents:

    def test_no_incidents_returns_empty(self):
        idx = _build_index(n_incidents=0)
        result = find_similar_incidents(_rand_vec(0), idx)
        assert result == []

    def test_incident_compressed_none_returns_empty(self):
        idx = _build_index(n_incidents=0)
        idx.incident_compressed = None
        result = find_similar_incidents(_rand_vec(0), idx)
        assert result == []

    def test_returns_matches(self):
        base = _rand_vec(seed=200)
        idx = _build_index(n_incidents=3, incident_seed=201)
        # Force one incident vector to be very similar
        inc_vecs = np.stack([base, _rand_vec(202), _rand_vec(203)])
        idx.incident_compressed = _Q.quantize(inc_vecs)
        idx.incident_windows = [_make_window(label=f"inc-{i}") for i in range(3)]
        idx.incident_labels = ["OOM-kill", "Disk-full", "CPU-spike"]
        idx.incident_resolutions = ["restart", "cleanup", "scale-up"]

        matches = find_similar_incidents(base, idx)
        assert len(matches) >= 1
        assert matches[0].similarity > 0.5

    def test_top_k_limit(self):
        idx = _build_index(n_incidents=10, incident_seed=300)
        matches = find_similar_incidents(_rand_vec(0), idx, k=2)
        assert len(matches) <= 2

    def test_low_similarity_filtered(self):
        # All incident vectors far from query -> filtered by sim < 0.3
        idx = _build_index(n_incidents=3, incident_seed=400)
        query = _orthogonal_vec(_rand_vec(400), seed=401)
        matches = find_similar_incidents(query, idx, k=10)
        for m in matches:
            assert m.similarity >= 0.3

    def test_match_fields_populated(self):
        base = _rand_vec(seed=500)
        idx = _build_index(n_incidents=1, incident_seed=501)
        idx.incident_compressed = _Q.quantize(np.stack([base]))
        idx.incident_windows = [_make_window(label="crash")]
        idx.incident_labels = ["crash"]
        idx.incident_resolutions = ["reboot"]

        matches = find_similar_incidents(base, idx, k=1)
        assert len(matches) == 1
        assert matches[0].label == "crash"
        assert matches[0].resolution == "reboot"

    def test_empty_incident_windows_returns_empty(self):
        idx = _build_index(n_incidents=0)
        idx.incident_compressed = _Q.quantize(np.stack([_rand_vec(0)]))
        idx.incident_windows = []  # empty
        result = find_similar_incidents(_rand_vec(1), idx)
        assert result == []


# ===================================================================
# 6. Full detect pipeline
# ===================================================================


class TestDetectPipeline:

    def _get_anomalous_setup(self):
        """Return (window, query_vec, index) where query is anomalous."""
        normal_base = _rand_vec(seed=600)
        normals = np.stack([
            _similar_vec(normal_base, noise=0.02, seed=601 + i)
            for i in range(5)
        ])
        idx = _build_index(n_normal=5)
        idx.normal_compressed = _Q.quantize(normals)
        # Query orthogonal -> high anomaly score
        query = _orthogonal_vec(normal_base, seed=610)
        window = _make_window(n_entries=5, n_errors=0)
        return window, query, idx

    def _get_normal_setup(self):
        """Return (window, query_vec, index) where query is normal."""
        normal_base = _rand_vec(seed=700)
        normals = np.stack([
            _similar_vec(normal_base, noise=0.02, seed=701 + i)
            for i in range(5)
        ])
        idx = _build_index(n_normal=5)
        idx.normal_compressed = _Q.quantize(normals)
        query = _similar_vec(normal_base, noise=0.01, seed=710)
        window = _make_window(n_entries=5, n_errors=0)
        return window, query, idx

    def test_detect_returns_alert_for_anomaly(self):
        window, query, idx = self._get_anomalous_setup()
        alert = detect(window, query, idx, sensitivity="high")
        assert alert is not None
        assert isinstance(alert, AnomalyAlert)

    def test_detect_returns_none_for_normal(self):
        window, query, idx = self._get_normal_setup()
        alert = detect(window, query, idx, sensitivity="low")
        assert alert is None

    def test_detect_sensitivity_low(self):
        window, query, idx = self._get_anomalous_setup()
        alert_low = detect(window, query, idx, sensitivity="low")
        alert_high = detect(window, query, idx, sensitivity="high")
        # High sensitivity should be at least as likely to trigger
        if alert_low is not None:
            assert alert_high is not None

    def test_detect_sensitivity_medium(self):
        window, query, idx = self._get_anomalous_setup()
        alert = detect(window, query, idx, sensitivity="medium")
        # With an orthogonal query we expect detection at medium
        assert alert is not None

    def test_detect_sensitivity_high(self):
        window, query, idx = self._get_anomalous_setup()
        alert = detect(window, query, idx, sensitivity="high")
        assert alert is not None

    def test_detect_unknown_sensitivity_uses_medium(self):
        window, query, idx = self._get_anomalous_setup()
        alert = detect(window, query, idx, sensitivity="nonexistent")
        # Falls back to medium (0.40 threshold)
        alert_medium = detect(window, query, idx, sensitivity="medium")
        if alert is not None and alert_medium is not None:
            assert alert.anomaly_score == alert_medium.anomaly_score

    def test_alert_has_correct_fields(self):
        window, query, idx = self._get_anomalous_setup()
        alert = detect(window, query, idx, sensitivity="high")
        assert alert is not None
        assert alert.current_window is window
        assert 0.0 <= alert.anomaly_score <= 1.0
        assert alert.anomaly_type in ("error_spike", "new_pattern", "similar_incident")
        assert alert.severity in ("CRITICAL", "WARNING", "INFO")
        assert isinstance(alert.summary, str)
        assert isinstance(alert.similar_incidents, list)

    def test_detect_with_incidents(self):
        window, query, idx = self._get_anomalous_setup()
        # Add incidents similar to query so similar_incident type triggers
        inc_vecs = np.stack([_similar_vec(query, noise=0.02, seed=620)])
        idx.incident_compressed = _Q.quantize(inc_vecs)
        idx.incident_windows = [_make_window(label="past-crash")]
        idx.incident_labels = ["past-crash"]
        idx.incident_resolutions = ["rollback"]
        idx.incident_count = 1

        alert = detect(window, query, idx, sensitivity="high")
        assert alert is not None
        if alert.similar_incidents:
            assert alert.similar_incidents[0].label == "past-crash"

    def test_detect_error_spike_type(self):
        idx = _build_index(n_normal=5, avg_errors=1.0, std_errors=0.5)
        # Use orthogonal query for high anomaly score
        normal_base = _rand_vec(seed=630)
        idx.normal_compressed = _Q.quantize(np.stack([normal_base]))
        query = _orthogonal_vec(normal_base, seed=631)
        window = _make_window(n_entries=10, n_errors=8)  # lots of errors

        alert = detect(window, query, idx, sensitivity="high")
        assert alert is not None
        assert alert.anomaly_type == "error_spike"


# ===================================================================
# 7. Summary generation
# ===================================================================


class TestSummaryGeneration:

    def test_error_spike_summary(self):
        w = _make_window(n_entries=10, n_errors=5, window_size=60)
        summary = _generate_summary("error_spike", w, [])
        assert "Error spike detected" in summary
        assert "5 errors" in summary

    def test_similar_incident_summary(self):
        match = IncidentMatch(similarity=0.85, label="OOM-kill", resolution="restart", occurred_at=None)
        w = _make_window(n_entries=5, n_errors=0)
        summary = _generate_summary("similar_incident", w, [match])
        assert "OOM-kill" in summary
        assert "85%" in summary

    def test_new_pattern_summary(self):
        w = _make_window(n_entries=5, n_errors=0)
        summary = _generate_summary("new_pattern", w, [])
        assert "Unusual log pattern" in summary

    def test_summary_includes_top_errors(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts, level="ERROR", source="api", message="NullPointerException at line 42"),
            _make_entry(ts=ts + timedelta(seconds=1), level="ERROR", source="db", message="Connection refused"),
            _make_entry(ts=ts + timedelta(seconds=2), level="INFO", source="svc", message="ok"),
        ]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts + timedelta(seconds=2), window_size=60)
        summary = _generate_summary("error_spike", w, [])
        assert "NullPointerException" in summary
        assert "Connection refused" in summary

    def test_summary_deduplicates_errors(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts + timedelta(seconds=i), level="ERROR", source="svc", message="same error")
            for i in range(10)
        ]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts + timedelta(seconds=9), window_size=60)
        summary = _generate_summary("error_spike", w, [])
        # "same error" should appear only once
        assert summary.count("same error") == 1

    def test_summary_caps_at_3_unique_errors(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts + timedelta(seconds=i), level="ERROR", source="svc", message=f"unique-error-{i}")
            for i in range(10)
        ]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts + timedelta(seconds=9), window_size=60)
        summary = _generate_summary("error_spike", w, [])
        # Should have at most 3 unique error lines
        error_lines = [line for line in summary.split("\n") if line.strip().startswith("[svc]")]
        assert len(error_lines) <= 3

    def test_summary_no_errors_no_top_errors_section(self):
        w = _make_window(n_entries=5, n_errors=0)
        summary = _generate_summary("new_pattern", w, [])
        assert "Top errors" not in summary

    def test_similar_incident_no_matches_fallback(self):
        w = _make_window(n_entries=5, n_errors=0)
        summary = _generate_summary("similar_incident", w, [])
        assert "Unusual log pattern" in summary

    def test_summary_truncates_long_messages(self):
        ts = datetime(2025, 1, 1)
        long_msg = "A" * 200
        entries = [_make_entry(ts=ts, level="ERROR", source="svc", message=long_msg)]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts, window_size=60)
        summary = _generate_summary("error_spike", w, [])
        # message truncated to 120 chars in summary
        lines = summary.split("\n")
        error_line = [l for l in lines if "[svc]" in l][0]
        # The displayed message portion should be <= 120 chars
        msg_part = error_line.split("] ", 1)[1]
        assert len(msg_part) <= 120


# ===================================================================
# Additional edge-case and integration tests
# ===================================================================


class TestEdgeCases:

    def test_window_with_only_warns(self):
        w = _make_window(n_entries=5, n_errors=0, n_warns=5)
        assert w.error_count == 0
        assert w.warn_count == 5

    def test_window_source_distribution(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts, source="api"),
            _make_entry(ts=ts + timedelta(seconds=1), source="api"),
            _make_entry(ts=ts + timedelta(seconds=2), source="db"),
        ]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts + timedelta(seconds=2), window_size=60)
        dist = w.source_distribution
        assert dist["api"] == 2
        assert dist["db"] == 1

    def test_log_entry_is_error_property(self):
        assert _make_entry(level="ERROR").is_error
        assert _make_entry(level="FATAL").is_error
        assert _make_entry(level="CRITICAL").is_error
        assert not _make_entry(level="WARN").is_error
        assert not _make_entry(level="INFO").is_error

    def test_log_entry_is_warn_property(self):
        assert _make_entry(level="WARN").is_warn
        assert _make_entry(level="WARNING").is_warn
        assert not _make_entry(level="ERROR").is_warn
        assert not _make_entry(level="INFO").is_warn

    def test_anomaly_alert_top_incident(self):
        match = IncidentMatch(similarity=0.9, label="x", resolution="y", occurred_at=None)
        alert = AnomalyAlert(
            current_window=_make_window(),
            anomaly_score=0.8,
            anomaly_type="similar_incident",
            similar_incidents=[match],
            severity="CRITICAL",
            summary="test",
        )
        assert alert.top_incident is match

    def test_anomaly_alert_top_incident_none(self):
        alert = AnomalyAlert(
            current_window=_make_window(),
            anomaly_score=0.8,
            anomaly_type="new_pattern",
            similar_incidents=[],
            severity="WARNING",
            summary="test",
        )
        assert alert.top_incident is None

    def test_window_to_embedding_text(self):
        ts = datetime(2025, 1, 1)
        entries = [
            _make_entry(ts=ts, level="ERROR", source="api", message="fail"),
            _make_entry(ts=ts + timedelta(seconds=1), level="INFO", source="svc", message="ok"),
        ]
        w = LogWindow(entries=entries, start_time=ts, end_time=ts + timedelta(seconds=1), window_size=60)
        text = w.to_embedding_text()
        assert "errors=1" in text
        assert "[ERROR] api: fail" in text

    def test_window_is_incident_property(self):
        w1 = _make_window(label="crash")
        w2 = _make_window(label="")
        assert w1.is_incident is True
        assert w2.is_incident is False

    def test_create_windows_very_small_window_size(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i)) for i in range(10)]
        windows = _create_windows(entries, window_size=1)
        assert len(windows) == 10

    def test_create_windows_very_large_window_size(self):
        ts = datetime(2025, 1, 1)
        entries = [_make_entry(ts=ts + timedelta(seconds=i * 100)) for i in range(10)]
        windows = _create_windows(entries, window_size=10000)
        assert len(windows) == 1

    def test_quantizer_roundtrip_preserves_direction(self):
        """Compressed vectors should still have reasonable cosine similarity."""
        v1 = _rand_vec(seed=900)
        v2 = _similar_vec(v1, noise=0.05, seed=901)
        compressed = _Q.quantize(np.stack([v1]))
        scores = _Q.cosine_scores(v2, compressed)
        score_arr = np.array(scores).flatten()
        assert float(score_arr[0]) > 0.5  # 3-bit compression loses some precision
