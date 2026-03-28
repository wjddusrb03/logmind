"""Stress and edge-case tests for LogMind collectors.

Covers FileCollector, StdinCollector, DockerCollector, and CommandCollector
with a focus on boundary conditions: large files, encoding issues, binary
content, glob patterns, permission errors, and empty inputs.

Run with:  pytest tests/test_collector_stress.py -v
"""

from __future__ import annotations

import os
import stat
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from logmind.collector import (
    CommandCollector,
    DockerCollector,
    FileCollector,
    StdinCollector,
)


# ===================================================================
# 1. FileCollector - basic operations
# ===================================================================

class TestFileCollectorBasic:
    """Basic FileCollector functionality."""

    def test_single_file(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3

    def test_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.log"
        f2 = tmp_path / "b.log"
        f1.write_text("alpha\n", encoding="utf-8")
        f2.write_text("bravo\n", encoding="utf-8")
        fc = FileCollector([str(f1), str(f2)])
        lines = fc.read_all()
        assert len(lines) == 2

    def test_files_sorted_order(self, tmp_path):
        """read_all reads files in sorted order."""
        f_b = tmp_path / "b.log"
        f_a = tmp_path / "a.log"
        f_b.write_text("BBB\n", encoding="utf-8")
        f_a.write_text("AAA\n", encoding="utf-8")
        fc = FileCollector([str(f_b), str(f_a)])
        lines = fc.read_all()
        assert lines[0].strip() == "AAA"
        assert lines[1].strip() == "BBB"

    def test_read_all_preserves_newlines(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("line1\nline2\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0] == "line1\n"

    def test_stream_yields_same_as_read_all(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("one\ntwo\nthree\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        streamed = list(fc.stream())
        batched = fc.read_all()
        assert len(streamed) == len(batched)


# ===================================================================
# 2. FileCollector - glob patterns
# ===================================================================

class TestFileCollectorGlob:
    """Glob expansion in FileCollector."""

    def test_star_glob(self, tmp_path):
        for name in ("svc1.log", "svc2.log", "svc3.log"):
            (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "svc*.log")])
        assert len(fc.files) == 3

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.log").write_text("root\n", encoding="utf-8")
        (sub / "child.log").write_text("child\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "**/*.log")])
        # should find at least the child
        assert any("child.log" in f for f in fc.files)

    def test_glob_no_match_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileCollector([str(tmp_path / "*.xyz_nonexistent")])

    def test_glob_mixed_with_explicit(self, tmp_path):
        explicit = tmp_path / "explicit.log"
        glob_target = tmp_path / "g1.log"
        explicit.write_text("e\n", encoding="utf-8")
        glob_target.write_text("g\n", encoding="utf-8")
        fc = FileCollector([str(explicit), str(tmp_path / "g*.log")])
        assert len(fc.files) >= 2


# ===================================================================
# 3. FileCollector - empty and edge cases
# ===================================================================

class TestFileCollectorEdgeCases:
    """Edge cases for FileCollector."""

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.log"
        f.write_text("", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines == []

    def test_single_line_no_newline(self, tmp_path):
        f = tmp_path / "one.log"
        f.write_text("single line no newline", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1
        assert lines[0] == "single line no newline"

    def test_very_long_line(self, tmp_path):
        f = tmp_path / "long.log"
        long_line = "X" * 100_000 + "\n"
        f.write_text(long_line, encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1
        assert len(lines[0]) == 100_001  # including newline

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            FileCollector(["/no/such/file.log"])

    def test_empty_paths_list_raises(self):
        with pytest.raises(FileNotFoundError):
            FileCollector([])

    def test_whitespace_only_lines(self, tmp_path):
        f = tmp_path / "ws.log"
        f.write_text("   \n\t\n\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3


# ===================================================================
# 4. FileCollector - large files (stress)
# ===================================================================

class TestFileCollectorLargeFiles:
    """Stress tests with larger files."""

    def test_ten_thousand_lines(self, tmp_path):
        f = tmp_path / "big.log"
        content = "".join(f"2025-06-15 10:00:{i%60:02d} INFO line {i}\n" for i in range(10_000))
        f.write_text(content, encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 10_000

    def test_hundred_files(self, tmp_path):
        for i in range(100):
            (tmp_path / f"shard_{i:03d}.log").write_text(
                f"line from shard {i}\n", encoding="utf-8"
            )
        fc = FileCollector([str(tmp_path / "shard_*.log")])
        assert len(fc.files) == 100
        lines = fc.read_all()
        assert len(lines) == 100

    def test_megabyte_file(self, tmp_path):
        f = tmp_path / "1mb.log"
        # ~1 MB of log lines
        line = "2025-06-15 10:00:00 INFO " + ("A" * 80) + "\n"
        count = (1024 * 1024) // len(line) + 1
        f.write_text(line * count, encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) >= count


# ===================================================================
# 5. FileCollector - encoding issues
# ===================================================================

class TestFileCollectorEncoding:
    """Encoding edge cases (errors='replace' behavior)."""

    def test_utf8_with_bom(self, tmp_path):
        f = tmp_path / "bom.log"
        f.write_bytes(b"\xef\xbb\xbfhello BOM\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1
        assert "hello BOM" in lines[0]

    def test_latin1_content(self, tmp_path):
        f = tmp_path / "latin.log"
        f.write_bytes(b"caf\xe9 latt\xe9\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1
        # errors='replace' replaces invalid bytes with U+FFFD
        assert len(lines[0]) > 0

    def test_null_bytes(self, tmp_path):
        f = tmp_path / "nulls.log"
        f.write_bytes(b"before\x00after\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1

    def test_mixed_encodings(self, tmp_path):
        f = tmp_path / "mixed.log"
        f.write_bytes(
            b"ascii line\n"
            b"\xc3\xa9 utf8 accent\n"
            b"\xff\xfe bad bytes\n"
        )
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3


# ===================================================================
# 6. FileCollector - binary content
# ===================================================================

class TestFileCollectorBinary:
    """Binary/garbage content."""

    def test_pure_binary(self, tmp_path):
        f = tmp_path / "binary.log"
        f.write_bytes(bytes(range(256)))
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        # Should not raise - errors='replace'
        assert isinstance(lines, list)

    def test_gzip_header_content(self, tmp_path):
        f = tmp_path / "compressed.log"
        f.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 100 + b"\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) >= 1

    def test_binary_interspersed_with_text(self, tmp_path):
        f = tmp_path / "messy.log"
        f.write_bytes(
            b"2025-06-15 10:00:00 INFO normal\n"
            b"\xff\xfe\xfd\xfc\n"
            b"2025-06-15 10:00:01 ERROR after binary\n"
        )
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3


# ===================================================================
# 7. FileCollector - permission errors
# ===================================================================

class TestFileCollectorPermissions:
    """Permission error handling."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod not effective on Windows for read removal"
    )
    def test_unreadable_file_skipped(self, tmp_path):
        f = tmp_path / "secret.log"
        f.write_text("classified\n", encoding="utf-8")
        f.chmod(0o000)
        try:
            fc = FileCollector([str(f)])
            lines = fc.read_all()
            # Should warn but not crash - returns empty
            assert lines == []
        finally:
            f.chmod(0o644)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod not effective on Windows for read removal"
    )
    def test_one_readable_one_not(self, tmp_path):
        good = tmp_path / "good.log"
        bad = tmp_path / "bad.log"
        good.write_text("readable\n", encoding="utf-8")
        bad.write_text("locked\n", encoding="utf-8")
        bad.chmod(0o000)
        try:
            fc = FileCollector([str(good), str(bad)])
            lines = fc.read_all()
            assert len(lines) == 1
            assert "readable" in lines[0]
        finally:
            bad.chmod(0o644)


# ===================================================================
# 8. FileCollector - stream mode
# ===================================================================

class TestFileCollectorStream:
    """Streaming (non-follow) mode tests."""

    def test_stream_basic(self, tmp_path):
        f = tmp_path / "s.log"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = list(fc.stream())
        assert len(lines) == 3

    def test_stream_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.log"
        f2 = tmp_path / "b.log"
        f1.write_text("1\n2\n", encoding="utf-8")
        f2.write_text("3\n4\n", encoding="utf-8")
        fc = FileCollector([str(f1), str(f2)])
        lines = list(fc.stream())
        assert len(lines) == 4

    def test_stream_empty_file(self, tmp_path):
        f = tmp_path / "empty.log"
        f.write_text("", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = list(fc.stream())
        assert lines == []

    def test_follow_flag_stored(self, tmp_path):
        f = tmp_path / "f.log"
        f.write_text("x\n", encoding="utf-8")
        fc = FileCollector([str(f)], follow=True)
        assert fc.follow is True

    def test_poll_interval_stored(self, tmp_path):
        f = tmp_path / "f.log"
        f.write_text("x\n", encoding="utf-8")
        fc = FileCollector([str(f)], poll_interval=2.0)
        assert fc.poll_interval == 2.0


# ===================================================================
# 9. StdinCollector
# ===================================================================

class TestStdinCollector:
    """StdinCollector tests using mocked stdin."""

    def test_stream_basic(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("line1\nline2\nline3\n"))
        sc = StdinCollector()
        lines = list(sc.stream())
        assert len(lines) == 3

    def test_read_all(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("alpha\nbeta\n"))
        sc = StdinCollector()
        lines = sc.read_all()
        assert len(lines) == 2

    def test_empty_stdin(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(""))
        sc = StdinCollector()
        lines = list(sc.stream())
        assert lines == []

    def test_single_line_stdin(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO("only one\n"))
        sc = StdinCollector()
        lines = sc.read_all()
        assert len(lines) == 1

    def test_large_stdin(self, monkeypatch):
        big_input = "".join(f"log line {i}\n" for i in range(5000))
        monkeypatch.setattr("sys.stdin", StringIO(big_input))
        sc = StdinCollector()
        lines = sc.read_all()
        assert len(lines) == 5000


# ===================================================================
# 10. DockerCollector - initialization and error handling
# ===================================================================

class TestDockerCollector:
    """DockerCollector initialization and basic tests (no real Docker)."""

    def test_init_basic(self):
        dc = DockerCollector("my-container")
        assert dc.container == "my-container"
        assert dc.follow is True
        assert dc.tail is None

    def test_init_with_options(self):
        dc = DockerCollector("web", follow=False, tail=100)
        assert dc.container == "web"
        assert dc.follow is False
        assert dc.tail == 100

    def test_stream_builds_correct_command(self):
        dc = DockerCollector("test-ctr", follow=True, tail=50)
        # We cannot run docker, but we can verify the object state
        assert dc.container == "test-ctr"
        assert dc.tail == 50

    @patch("subprocess.Popen")
    def test_stream_calls_docker_logs(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["line1\n", "line2\n"])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("app", follow=True, tail=10)
        lines = list(dc.stream())
        assert len(lines) == 2

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "docker" in cmd
        assert "logs" in cmd
        assert "--follow" in cmd
        assert "--tail" in cmd
        assert "app" in cmd

    @patch("subprocess.run")
    def test_read_all_calls_docker_logs(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="out1\nout2\n", stderr=""
        )
        dc = DockerCollector("app", tail=5)
        lines = dc.read_all()
        assert len(lines) == 2

    @patch("subprocess.Popen")
    def test_stream_without_follow(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["log\n"])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("ctr", follow=False)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert "--follow" not in cmd


# ===================================================================
# 11. CommandCollector - initialization and error handling
# ===================================================================

class TestCommandCollector:
    """CommandCollector tests (mocked subprocess)."""

    def test_init_basic(self):
        cc = CommandCollector("journalctl -f")
        assert cc.command == "journalctl -f"
        assert cc.follow is True

    def test_init_follow_false(self):
        cc = CommandCollector("cat /var/log/syslog", follow=False)
        assert cc.follow is False

    @patch("subprocess.Popen")
    def test_stream(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["msg1\n", "msg2\n", "msg3\n"])
        mock_popen.return_value = mock_proc

        cc = CommandCollector("tail -f /var/log/app.log")
        lines = list(cc.stream())
        assert len(lines) == 3
        mock_popen.assert_called_once()
        assert mock_popen.call_args[1]["shell"] is True

    @patch("subprocess.run")
    def test_read_all(self, mock_run):
        mock_run.return_value = MagicMock(stdout="hello\nworld\n", stderr="")
        cc = CommandCollector("echo hello")
        lines = cc.read_all()
        assert len(lines) == 2

    @patch("subprocess.run")
    def test_read_all_combines_stdout_stderr(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="out\n", stderr="err\n"
        )
        cc = CommandCollector("mixed")
        lines = cc.read_all()
        assert len(lines) == 2

    @patch("subprocess.Popen")
    def test_stream_terminates_process(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["x\n"])
        mock_popen.return_value = mock_proc

        cc = CommandCollector("cmd")
        list(cc.stream())
        mock_proc.terminate.assert_called_once()

    @patch("subprocess.run")
    def test_read_all_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="")
        cc = CommandCollector("true")
        lines = cc.read_all()
        assert lines == [] or lines == [""]  # splitlines may be empty


# ===================================================================
# 12. Cross-collector edge cases
# ===================================================================

class TestCrossCollectorEdgeCases:
    """Edge cases spanning multiple collector types."""

    def test_file_collector_with_follow_false_ends(self, tmp_path):
        """Non-follow stream should terminate after reading all content."""
        f = tmp_path / "done.log"
        f.write_text("a\nb\n", encoding="utf-8")
        fc = FileCollector([str(f)], follow=False)
        lines = list(fc.stream())
        assert len(lines) == 2

    def test_file_collector_handles_crlf(self, tmp_path):
        f = tmp_path / "win.log"
        f.write_bytes(b"line1\r\nline2\r\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 2

    def test_file_collector_handles_mixed_line_endings(self, tmp_path):
        f = tmp_path / "mixed_eol.log"
        f.write_bytes(b"unix\nwindows\r\nold_mac\rend\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        # Python universal newlines should split all
        assert len(lines) >= 3

    def test_file_collector_unicode_filename(self, tmp_path):
        f = tmp_path / "app_\ub85c\uadf8.log"
        f.write_text("korean filename\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1

    def test_file_collector_special_characters_in_content(self, tmp_path):
        f = tmp_path / "special.log"
        f.write_text(
            "2025-06-15 10:00:00 INFO emoji: \U0001f525\U0001f680\n"
            "2025-06-15 10:00:01 ERROR tab\there\n"
            "2025-06-15 10:00:02 WARN backslash \\\\ path\n",
            encoding="utf-8",
        )
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3
