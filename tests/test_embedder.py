"""Tests for embedder module."""

import pytest
from datetime import datetime
from logmind.models import LogEntry, LogWindow
from logmind.embedder import _create_windows


class TestCreateWindows:
    def _make_entries(self, count, start_second=0, interval=5):
        from datetime import timedelta
        entries = []
        base = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(count):
            ts = base + timedelta(seconds=start_second + i * interval)
            entries.append(LogEntry(
                timestamp=ts,
                level="INFO" if i % 3 != 0 else "ERROR",
                source=f"app{i % 3}",
                message=f"message {i}",
                raw=f"raw {i}",
            ))
        return entries

    def test_single_window(self):
        entries = self._make_entries(5, interval=5)  # 25 seconds total
        windows = _create_windows(entries, window_size=60)
        assert len(windows) == 1
        assert windows[0].total_count == 5

    def test_multiple_windows(self):
        entries = self._make_entries(24, interval=5)  # 120 seconds total
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 2

    def test_empty_entries(self):
        windows = _create_windows([])
        assert windows == []

    def test_no_timestamps(self):
        entries = [
            LogEntry(None, "INFO", "app", f"msg{i}", f"raw{i}")
            for i in range(20)
        ]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 1
        assert all(w.total_count > 0 for w in windows)

    def test_window_size_respected(self):
        entries = self._make_entries(60, interval=2)  # 120 seconds
        windows = _create_windows(entries, window_size=30)
        assert len(windows) >= 3
