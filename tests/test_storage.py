"""Tests for storage module."""

import pytest
from logmind.storage import save_index, load_index, index_exists
from logmind.models import LogMindIndex


def _make_index():
    from langchain_turboquant import TurboQuantizer
    import numpy as np
    dim = 32
    quantizer = TurboQuantizer(dim=dim, bits=3)
    vecs = np.random.randn(5, dim).astype(np.float32)
    compressed = quantizer.quantize(vecs)
    return LogMindIndex(
        normal_compressed=compressed,
        normal_windows=[],
        incident_compressed=None,
        incident_windows=[],
        incident_labels=[],
        incident_resolutions=[],
        quantizer=quantizer,
        model_name="test",
        embedding_dim=dim,
        total_lines=50,
        total_windows=5,
        incident_count=0,
        error_count=3,
        warn_count=2,
        sources=["app"],
        learn_time=0.5,
    )


class TestStorage:
    def test_save_and_load(self, tmp_path):
        index = _make_index()
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.total_lines == 50
        assert loaded.model_name == "test"

    def test_load_missing(self, tmp_path):
        result = load_index(str(tmp_path))
        assert result is None

    def test_index_exists(self, tmp_path):
        assert index_exists(str(tmp_path)) is False
        index = _make_index()
        save_index(index, str(tmp_path))
        assert index_exists(str(tmp_path)) is True
