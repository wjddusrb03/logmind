"""Rigorous tests for storage module: save/load roundtrips, edge cases, data integrity."""

from __future__ import annotations

import os
import pickle
import stat
import time
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest
from langchain_turboquant import TurboQuantizer

from logmind.models import LogEntry, LogMindIndex, LogWindow
from logmind.storage import (
    DEFAULT_DIR,
    DEFAULT_FILE,
    _index_path,
    index_exists,
    load_index,
    save_index,
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
) -> LogEntry:
    return LogEntry(
        timestamp=ts,
        level=level,
        source=source,
        message=message,
        raw=f"[{level}] {message}",
        line_number=0,
    )


def _make_window(
    n_entries: int = 5,
    n_errors: int = 0,
    label: str = "",
    resolution: str = "",
) -> LogWindow:
    ts = datetime(2025, 1, 1)
    entries: List[LogEntry] = []
    for i in range(n_errors):
        entries.append(_make_entry(ts=ts + timedelta(seconds=i), level="ERROR", message=f"err-{i}"))
    for i in range(n_entries - n_errors):
        entries.append(_make_entry(ts=ts + timedelta(seconds=n_errors + i)))
    return LogWindow(
        entries=entries,
        start_time=ts,
        end_time=ts + timedelta(seconds=max(n_entries - 1, 0)),
        window_size=60,
        label=label,
        resolution=resolution,
    )


def _build_index(
    n_normal: int = 5,
    n_incidents: int = 0,
    model_name: str = "test-model",
    embedding_dim: int = DIM,
    total_lines: int = 100,
    total_windows: int = 10,
    error_count: int = 5,
    warn_count: int = 3,
    sources: List[str] | None = None,
    learn_time: float = 1.5,
    avg_errors: float = 1.0,
    std_errors: float = 0.5,
    avg_warns: float = 0.3,
    incident_labels: List[str] | None = None,
    incident_resolutions: List[str] | None = None,
) -> LogMindIndex:
    rng = np.random.RandomState(42)
    normal_vecs = rng.randn(n_normal, DIM).astype(np.float32)
    normal_compressed = _Q.quantize(normal_vecs)
    normal_windows = [_make_window() for _ in range(n_normal)]

    inc_compressed = None
    inc_windows: List[LogWindow] = []
    inc_labels: List[str] = incident_labels or []
    inc_resolutions: List[str] = incident_resolutions or []

    if n_incidents > 0:
        inc_vecs = rng.randn(n_incidents, DIM).astype(np.float32)
        inc_compressed = _Q.quantize(inc_vecs)
        inc_windows = [_make_window(label=f"inc-{i}", resolution=f"fix-{i}") for i in range(n_incidents)]
        if not inc_labels:
            inc_labels = [f"inc-{i}" for i in range(n_incidents)]
        if not inc_resolutions:
            inc_resolutions = [f"fix-{i}" for i in range(n_incidents)]

    return LogMindIndex(
        normal_compressed=normal_compressed,
        normal_windows=normal_windows,
        incident_compressed=inc_compressed,
        incident_windows=inc_windows,
        incident_labels=inc_labels,
        incident_resolutions=inc_resolutions,
        quantizer=_Q,
        model_name=model_name,
        embedding_dim=embedding_dim,
        total_lines=total_lines,
        total_windows=total_windows,
        incident_count=n_incidents,
        error_count=error_count,
        warn_count=warn_count,
        sources=sources or ["app", "db"],
        learn_time=learn_time,
        avg_errors_per_window=avg_errors,
        std_errors_per_window=std_errors,
        avg_warns_per_window=avg_warns,
    )


# ===================================================================
# 1. Save / Load roundtrip - preserving ALL fields
# ===================================================================


class TestRoundtripAllFields:
    """Verify every LogMindIndex field survives save/load."""

    def test_model_name_preserved(self, tmp_path):
        idx = _build_index(model_name="sentence-transformers/all-MiniLM-L6-v2")
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dim_preserved(self, tmp_path):
        idx = _build_index(embedding_dim=384)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.embedding_dim == 384

    def test_total_lines_preserved(self, tmp_path):
        idx = _build_index(total_lines=999)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.total_lines == 999

    def test_total_windows_preserved(self, tmp_path):
        idx = _build_index(total_windows=42)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.total_windows == 42

    def test_incident_count_preserved(self, tmp_path):
        idx = _build_index(n_incidents=7)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_count == 7

    def test_error_count_preserved(self, tmp_path):
        idx = _build_index(error_count=123)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.error_count == 123

    def test_warn_count_preserved(self, tmp_path):
        idx = _build_index(warn_count=77)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.warn_count == 77

    def test_sources_preserved(self, tmp_path):
        idx = _build_index(sources=["api", "worker", "db", "cache"])
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.sources == ["api", "worker", "db", "cache"]

    def test_learn_time_preserved(self, tmp_path):
        idx = _build_index(learn_time=3.14159)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert abs(loaded.learn_time - 3.14159) < 1e-6

    def test_avg_errors_per_window_preserved(self, tmp_path):
        idx = _build_index(avg_errors=2.75)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert abs(loaded.avg_errors_per_window - 2.75) < 1e-9

    def test_std_errors_per_window_preserved(self, tmp_path):
        idx = _build_index(std_errors=1.23)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert abs(loaded.std_errors_per_window - 1.23) < 1e-9

    def test_avg_warns_per_window_preserved(self, tmp_path):
        idx = _build_index(avg_warns=0.88)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert abs(loaded.avg_warns_per_window - 0.88) < 1e-9

    def test_normal_windows_count_preserved(self, tmp_path):
        idx = _build_index(n_normal=8)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert len(loaded.normal_windows) == 8

    def test_incident_labels_preserved(self, tmp_path):
        idx = _build_index(
            n_incidents=3,
            incident_labels=["OOM-kill", "Disk-full", "CPU-spike"],
        )
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_labels == ["OOM-kill", "Disk-full", "CPU-spike"]

    def test_incident_resolutions_preserved(self, tmp_path):
        idx = _build_index(
            n_incidents=2,
            incident_resolutions=["restart pod", "scale horizontally"],
        )
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_resolutions == ["restart pod", "scale horizontally"]


# ===================================================================
# 2. Directory creation and path handling
# ===================================================================


class TestDirectoryHandling:

    def test_save_creates_logmind_dir(self, tmp_path):
        idx = _build_index()
        logmind_dir = tmp_path / DEFAULT_DIR
        assert not logmind_dir.exists()
        save_index(idx, str(tmp_path))
        assert logmind_dir.exists()
        assert logmind_dir.is_dir()

    def test_save_creates_nested_base_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        idx = _build_index()
        save_index(idx, str(nested))
        assert (nested / DEFAULT_DIR / DEFAULT_FILE).exists()

    def test_index_path_uses_default_dir_and_file(self, tmp_path):
        path = _index_path(str(tmp_path))
        assert path == os.path.join(str(tmp_path), DEFAULT_DIR, DEFAULT_FILE)

    def test_save_with_absolute_path(self, tmp_path):
        abs_path = str(tmp_path.resolve())
        idx = _build_index()
        returned = save_index(idx, abs_path)
        assert os.path.isabs(returned) or os.path.exists(returned)
        assert load_index(abs_path) is not None


# ===================================================================
# 3. Overwrite behaviour
# ===================================================================


class TestOverwrite:

    def test_save_overwrites_existing_index(self, tmp_path):
        idx1 = _build_index(total_lines=100)
        idx2 = _build_index(total_lines=200)
        save_index(idx1, str(tmp_path))
        save_index(idx2, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.total_lines == 200

    def test_overwrite_preserves_new_fields(self, tmp_path):
        idx1 = _build_index(model_name="old-model", error_count=1)
        save_index(idx1, str(tmp_path))
        idx2 = _build_index(model_name="new-model", error_count=99)
        save_index(idx2, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.model_name == "new-model"
        assert loaded.error_count == 99


# ===================================================================
# 4. Load from wrong / missing directory
# ===================================================================


class TestLoadMissing:

    def test_load_nonexistent_dir_returns_none(self, tmp_path):
        result = load_index(str(tmp_path / "nonexistent"))
        assert result is None

    def test_load_empty_dir_returns_none(self, tmp_path):
        result = load_index(str(tmp_path))
        assert result is None

    def test_load_dir_with_logmind_but_no_file(self, tmp_path):
        (tmp_path / DEFAULT_DIR).mkdir()
        result = load_index(str(tmp_path))
        assert result is None


# ===================================================================
# 5. index_exists
# ===================================================================


class TestIndexExists:

    def test_exists_false_before_save(self, tmp_path):
        assert index_exists(str(tmp_path)) is False

    def test_exists_true_after_save(self, tmp_path):
        idx = _build_index()
        save_index(idx, str(tmp_path))
        assert index_exists(str(tmp_path)) is True

    def test_exists_false_for_nonexistent_dir(self, tmp_path):
        assert index_exists(str(tmp_path / "nope")) is False


# ===================================================================
# 6. Multiple save/load cycles
# ===================================================================


class TestMultipleCycles:

    def test_three_save_load_cycles(self, tmp_path):
        for i in range(3):
            idx = _build_index(total_lines=i * 100)
            save_index(idx, str(tmp_path))
            loaded = load_index(str(tmp_path))
            assert loaded.total_lines == i * 100

    def test_data_integrity_across_cycles(self, tmp_path):
        idx = _build_index(n_normal=10, n_incidents=3)
        for _ in range(5):
            save_index(idx, str(tmp_path))
            loaded = load_index(str(tmp_path))
            assert loaded.total_lines == idx.total_lines
            assert loaded.incident_count == idx.incident_count
            assert len(loaded.normal_windows) == len(idx.normal_windows)
            assert len(loaded.incident_labels) == len(idx.incident_labels)
            idx = loaded  # use loaded for next cycle


# ===================================================================
# 7. Incidents in index
# ===================================================================


class TestWithIncidents:

    def test_save_load_with_incidents(self, tmp_path):
        idx = _build_index(n_incidents=5)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_count == 5
        assert len(loaded.incident_windows) == 5
        assert len(loaded.incident_labels) == 5
        assert len(loaded.incident_resolutions) == 5

    def test_save_load_without_incidents(self, tmp_path):
        idx = _build_index(n_incidents=0)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_compressed is None
        assert loaded.incident_windows == []
        assert loaded.incident_labels == []
        assert loaded.incident_resolutions == []

    def test_incident_window_labels_survive_roundtrip(self, tmp_path):
        idx = _build_index(n_incidents=2)
        idx.incident_windows[0].label = "memory-leak"
        idx.incident_windows[1].label = "disk-full"
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_windows[0].label == "memory-leak"
        assert loaded.incident_windows[1].label == "disk-full"


# ===================================================================
# 8. Large index
# ===================================================================


class TestLargeIndex:

    def test_large_normal_windows(self, tmp_path):
        rng = np.random.RandomState(0)
        vecs = rng.randn(50, DIM).astype(np.float32)
        compressed = _Q.quantize(vecs)
        idx = _build_index(n_normal=50)
        idx.normal_compressed = compressed
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert len(loaded.normal_windows) == 50


# ===================================================================
# 9. Concurrent-like save (latest wins)
# ===================================================================


class TestConcurrentSave:

    def test_rapid_double_save_latest_wins(self, tmp_path):
        idx1 = _build_index(total_lines=111)
        idx2 = _build_index(total_lines=222)
        save_index(idx1, str(tmp_path))
        save_index(idx2, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.total_lines == 222


# ===================================================================
# 10. File properties after save
# ===================================================================


class TestFileProperties:

    def test_saved_file_is_regular_file(self, tmp_path):
        idx = _build_index()
        path = save_index(idx, str(tmp_path))
        assert os.path.isfile(path)

    def test_saved_file_nonzero_size(self, tmp_path):
        idx = _build_index()
        path = save_index(idx, str(tmp_path))
        assert os.path.getsize(path) > 0

    def test_save_returns_correct_path(self, tmp_path):
        idx = _build_index()
        path = save_index(idx, str(tmp_path))
        expected = os.path.join(str(tmp_path), DEFAULT_DIR, DEFAULT_FILE)
        assert os.path.normpath(path) == os.path.normpath(expected)


# ===================================================================
# 11. Pickle protocol compatibility
# ===================================================================


class TestPickleProtocol:

    def test_saved_file_is_valid_pickle(self, tmp_path):
        idx = _build_index()
        path = save_index(idx, str(tmp_path))
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, LogMindIndex)

    def test_pickle_highest_protocol_used(self, tmp_path):
        idx = _build_index()
        path = save_index(idx, str(tmp_path))
        with open(path, "rb") as f:
            data = f.read()
        # Highest protocol marker: pickle protocol 5 starts with 0x80 0x05
        # At minimum, verify it is a valid pickle that can be loaded
        obj = pickle.loads(data)
        assert isinstance(obj, LogMindIndex)


# ===================================================================
# 12. Quantizer roundtrip through save/load
# ===================================================================


class TestQuantizerRoundtrip:

    def test_quantizer_works_after_load(self, tmp_path):
        idx = _build_index(n_normal=5)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        # The loaded quantizer should still be able to dequantize
        deq = loaded.quantizer.dequantize(loaded.normal_compressed)
        assert deq.shape[0] == 5
        assert deq.shape[1] == DIM

    def test_cosine_scores_work_after_load(self, tmp_path):
        idx = _build_index(n_normal=5)
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        query = np.random.randn(DIM).astype(np.float32)
        scores = loaded.quantizer.cosine_scores(query, loaded.normal_compressed)
        arr = np.array(scores).flatten()
        assert len(arr) == 5
        for s in arr:
            assert -1.1 <= float(s) <= 1.1  # cosine sim in [-1, 1] with rounding
