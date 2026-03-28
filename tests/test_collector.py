"""Tests for log collectors."""

import pytest
from logmind.collector import FileCollector, StdinCollector


class TestFileCollector:
    def test_read_single_file(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("line1\nline2\nline3\n")
        collector = FileCollector([str(log)])
        lines = collector.read_all()
        assert len(lines) == 3

    def test_read_multiple_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"test{i}.log").write_text(f"log{i}\n")
        collector = FileCollector([str(tmp_path / "*.log")])
        lines = collector.read_all()
        assert len(lines) == 3

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            FileCollector(["/nonexistent/path/log.txt"])

    def test_stream_no_follow(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("line1\nline2\n")
        collector = FileCollector([str(log)], follow=False)
        lines = list(collector.stream())
        assert len(lines) == 2

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "app.log").write_text("line1\n")
        (tmp_path / "error.log").write_text("line2\n")
        (tmp_path / "other.txt").write_text("skip\n")
        collector = FileCollector([str(tmp_path / "*.log")])
        assert len(collector.files) == 2


class TestStdinCollector:
    def test_instance(self):
        collector = StdinCollector()
        assert hasattr(collector, "stream")
        assert hasattr(collector, "read_all")
