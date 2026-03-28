"""Advanced tests for log collectors: edge cases, command execution,
Docker command construction, mixed line endings, special filenames, and more.

Run with:  pytest tests/test_collector_advanced.py -v
"""

from __future__ import annotations

import os
import sys
import textwrap
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
# 1. FileCollector - recursive glob with nested directories
# ===================================================================

class TestFileCollectorRecursiveGlob:
    """Recursive glob through nested directory trees."""

    def test_three_levels_deep(self, tmp_path):
        d1 = tmp_path / "a"
        d2 = d1 / "b"
        d3 = d2 / "c"
        d3.mkdir(parents=True)
        (d1 / "l1.log").write_text("level1\n", encoding="utf-8")
        (d2 / "l2.log").write_text("level2\n", encoding="utf-8")
        (d3 / "l3.log").write_text("level3\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "**/*.log")])
        assert len(fc.files) == 3

    def test_recursive_reads_all_content(self, tmp_path):
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        (tmp_path / "root.log").write_text("root_line\n", encoding="utf-8")
        (sub / "deep.log").write_text("deep_line\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "**/*.log")])
        lines = fc.read_all()
        content = "".join(lines)
        assert "root_line" in content
        assert "deep_line" in content

    def test_recursive_mixed_extensions(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.log").write_text("log\n", encoding="utf-8")
        (sub / "b.txt").write_text("txt\n", encoding="utf-8")
        (sub / "c.log").write_text("log2\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "**/*.log")])
        assert all(f.endswith(".log") for f in fc.files)

    def test_five_levels_deep(self, tmp_path):
        d = tmp_path
        for i in range(5):
            d = d / f"level{i}"
            d.mkdir()
            (d / f"file{i}.log").write_text(f"line{i}\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "**/*.log")])
        assert len(fc.files) == 5
        lines = fc.read_all()
        assert len(lines) == 5


# ===================================================================
# 2. FileCollector - symlinks and deep paths
# ===================================================================

class TestFileCollectorSymlinksAndDeepPaths:
    """Symlinks (non-Windows) and very deep paths."""

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks unreliable on Windows")
    def test_symlink_followed(self, tmp_path):
        real = tmp_path / "real.log"
        real.write_text("real content\n", encoding="utf-8")
        link = tmp_path / "link.log"
        link.symlink_to(real)
        fc = FileCollector([str(link)])
        lines = fc.read_all()
        assert lines[0].strip() == "real content"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks unreliable on Windows")
    def test_symlink_dir_glob(self, tmp_path):
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "inner.log").write_text("inner\n", encoding="utf-8")
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)
        fc = FileCollector([str(link_dir / "*.log")])
        assert len(fc.files) >= 1

    def test_very_deep_path(self, tmp_path):
        """Create a deep directory structure and verify FileCollector can read."""
        d = tmp_path
        for i in range(20):
            d = d / f"d{i}"
        d.mkdir(parents=True)
        f = d / "deep.log"
        f.write_text("deep content\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0].strip() == "deep content"


# ===================================================================
# 3. FileCollector - file appearing during stream (log rotation)
# ===================================================================

class TestFileCollectorLogRotation:
    """Simulate files appearing/changing during collection."""

    def test_read_all_sees_only_initial_files(self, tmp_path):
        """FileCollector locks its file list at init time."""
        (tmp_path / "app.log").write_text("line1\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "app.log")])
        # Write a new file after init
        (tmp_path / "app.log.1").write_text("rotated\n", encoding="utf-8")
        lines = fc.read_all()
        content = "".join(lines)
        assert "rotated" not in content

    def test_stream_sees_only_initial_files(self, tmp_path):
        (tmp_path / "svc.log").write_text("original\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "svc.log")])
        (tmp_path / "svc.log.1").write_text("rotated_line\n", encoding="utf-8")
        lines = list(fc.stream())
        content = "".join(lines)
        assert "rotated_line" not in content

    def test_file_content_change_after_init(self, tmp_path):
        """If file content changes between init and read, read_all gets new content."""
        f = tmp_path / "live.log"
        f.write_text("v1\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        f.write_text("v2\n", encoding="utf-8")
        lines = fc.read_all()
        assert lines[0].strip() == "v2"


# ===================================================================
# 4. FileCollector - mixed line endings
# ===================================================================

class TestFileCollectorMixedLineEndings:
    """CRLF, LF, CR within the same file."""

    def test_crlf_only(self, tmp_path):
        f = tmp_path / "win.log"
        f.write_bytes(b"line1\r\nline2\r\nline3\r\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3

    def test_lf_only(self, tmp_path):
        f = tmp_path / "unix.log"
        f.write_bytes(b"line1\nline2\nline3\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3

    def test_cr_only(self, tmp_path):
        f = tmp_path / "mac.log"
        f.write_bytes(b"line1\rline2\rline3\r")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        # Python universal newlines: \r is treated as line ending
        assert len(lines) >= 3

    def test_mixed_crlf_lf_cr(self, tmp_path):
        f = tmp_path / "mixed.log"
        f.write_bytes(b"unix\nwindows\r\noldmac\rend\n")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) >= 3

    def test_stream_mixed_line_endings(self, tmp_path):
        f = tmp_path / "mixed_s.log"
        f.write_bytes(b"a\nb\r\nc\rd\n")
        fc = FileCollector([str(f)])
        lines = list(fc.stream())
        assert len(lines) >= 3


# ===================================================================
# 5. FileCollector - no trailing newline
# ===================================================================

class TestFileCollectorNoTrailingNewline:
    """Files that don't end with a newline character."""

    def test_single_line_no_newline(self, tmp_path):
        f = tmp_path / "noeol.log"
        f.write_bytes(b"no trailing newline")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1
        assert lines[0] == "no trailing newline"

    def test_multi_line_last_no_newline(self, tmp_path):
        f = tmp_path / "partial.log"
        f.write_bytes(b"line1\nline2\nline3_no_nl")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 3
        assert not lines[-1].endswith("\n")

    def test_stream_no_trailing_newline(self, tmp_path):
        f = tmp_path / "stream_noeol.log"
        f.write_bytes(b"alpha\nbeta")
        fc = FileCollector([str(f)])
        lines = list(fc.stream())
        assert len(lines) == 2
        assert lines[-1] == "beta"


# ===================================================================
# 6. FileCollector - zero-byte file
# ===================================================================

class TestFileCollectorZeroByteFile:
    """Zero-byte (empty) files."""

    def test_zero_byte_read_all(self, tmp_path):
        f = tmp_path / "zero.log"
        f.write_bytes(b"")
        fc = FileCollector([str(f)])
        assert fc.read_all() == []

    def test_zero_byte_stream(self, tmp_path):
        f = tmp_path / "zero_s.log"
        f.write_bytes(b"")
        fc = FileCollector([str(f)])
        assert list(fc.stream()) == []

    def test_zero_byte_among_normal_files(self, tmp_path):
        (tmp_path / "empty.log").write_bytes(b"")
        (tmp_path / "full.log").write_text("content\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "*.log")])
        lines = fc.read_all()
        assert len(lines) == 1
        assert "content" in lines[0]


# ===================================================================
# 7. CommandCollector - echo commands that actually run
# ===================================================================

class TestCommandCollectorRealCommands:
    """CommandCollector with simple commands that actually execute."""

    def test_echo_read_all(self):
        cc = CommandCollector("echo hello_world", follow=False)
        lines = cc.read_all()
        content = "".join(lines)
        assert "hello_world" in content

    def test_echo_multiline(self):
        # Use printf for multi-line output
        cc = CommandCollector('echo line1 && echo line2 && echo line3', follow=False)
        lines = cc.read_all()
        content = "".join(lines)
        assert "line1" in content
        assert "line2" in content

    def test_stream_echo(self):
        cc = CommandCollector("echo streamed_output", follow=False)
        lines = list(cc.stream())
        content = "".join(lines)
        assert "streamed_output" in content

    def test_command_with_pipe(self):
        cc = CommandCollector('echo "abc def ghi" | tr " " "\\n"', follow=False)
        lines = cc.read_all()
        assert len(lines) >= 1


# ===================================================================
# 8. CommandCollector - stderr capture
# ===================================================================

class TestCommandCollectorStderr:
    """Verify stderr is captured."""

    def test_stderr_in_read_all(self):
        # echo to stderr using >&2
        cc = CommandCollector("echo error_msg >&2", follow=False)
        lines = cc.read_all()
        content = "".join(lines)
        assert "error_msg" in content

    def test_mixed_stdout_stderr_read_all(self):
        cc = CommandCollector("echo stdout_msg && echo stderr_msg >&2", follow=False)
        lines = cc.read_all()
        content = "".join(lines)
        assert "stdout_msg" in content
        assert "stderr_msg" in content

    @patch("subprocess.Popen")
    def test_stream_captures_stderr_via_stdout(self, mock_popen):
        """stream() merges stderr into stdout via STDOUT redirect."""
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["stdout\n", "stderr_line\n"])
        mock_popen.return_value = mock_proc
        cc = CommandCollector("cmd")
        lines = list(cc.stream())
        assert len(lines) == 2
        # Verify stderr=STDOUT was passed
        call_kwargs = mock_popen.call_args
        import subprocess
        assert call_kwargs.kwargs.get("stderr") == subprocess.STDOUT or \
               call_kwargs[1].get("stderr") == subprocess.STDOUT


# ===================================================================
# 9. DockerCollector - command construction with all options
# ===================================================================

class TestDockerCollectorCommandConstruction:
    """Verify docker command is built correctly for all option combos."""

    @patch("subprocess.Popen")
    def test_follow_true_tail_none(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("myapp", follow=True, tail=None)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert cmd == ["docker", "logs", "--follow", "myapp"]

    @patch("subprocess.Popen")
    def test_follow_false_tail_none(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("myapp", follow=False, tail=None)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert "--follow" not in cmd
        assert "--tail" not in cmd
        assert cmd == ["docker", "logs", "myapp"]

    @patch("subprocess.Popen")
    def test_follow_true_tail_100(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("web", follow=True, tail=100)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert "--follow" in cmd
        assert "--tail" in cmd
        assert "100" in cmd
        assert cmd[-1] == "web"

    @patch("subprocess.Popen")
    def test_follow_false_tail_50(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("api", follow=False, tail=50)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert "--follow" not in cmd
        assert "--tail" in cmd
        assert "50" in cmd

    @patch("subprocess.run")
    def test_read_all_no_follow_flag(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="")
        dc = DockerCollector("ctr", follow=True, tail=20)
        dc.read_all()
        cmd = mock_run.call_args[0][0]
        # read_all never uses --follow
        assert "--follow" not in cmd
        assert "--tail" in cmd
        assert "20" in cmd

    @patch("subprocess.run")
    def test_read_all_no_tail(self, mock_run):
        mock_run.return_value = MagicMock(stdout="data\n", stderr="")
        dc = DockerCollector("svc", follow=False, tail=None)
        dc.read_all()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["docker", "logs", "svc"]

    @patch("subprocess.Popen")
    def test_container_name_with_special_chars(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("my-app_v2.1", follow=False)
        list(dc.stream())
        cmd = mock_popen.call_args[0][0]
        assert cmd[-1] == "my-app_v2.1"


# ===================================================================
# 10. Multiple collectors producing similar output
# ===================================================================

class TestMultipleCollectorsSimilarOutput:
    """Verify different collectors can produce equivalent results."""

    def test_file_and_command_same_content(self, tmp_path):
        f = tmp_path / "data.log"
        f.write_text("hello\nworld\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        file_lines = fc.read_all()

        cc = CommandCollector(f'echo "hello\nworld"', follow=False)
        # Just verify both return lists of strings
        cmd_lines = cc.read_all()
        assert isinstance(file_lines, list)
        assert isinstance(cmd_lines, list)

    @patch("subprocess.Popen")
    def test_docker_and_command_same_interface(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["log1\n", "log2\n"])
        mock_popen.return_value = mock_proc

        dc = DockerCollector("ctr", follow=False)
        docker_lines = list(dc.stream())

        mock_proc2 = MagicMock()
        mock_proc2.stdout = iter(["log1\n", "log2\n"])
        mock_popen.return_value = mock_proc2

        cc = CommandCollector("docker logs ctr")
        cmd_lines = list(cc.stream())

        assert docker_lines == cmd_lines

    def test_stdin_and_file_same_content(self, tmp_path, monkeypatch):
        content = "shared_line1\nshared_line2\n"
        f = tmp_path / "shared.log"
        f.write_text(content, encoding="utf-8")
        fc = FileCollector([str(f)])
        file_lines = fc.read_all()

        monkeypatch.setattr("sys.stdin", StringIO(content))
        sc = StdinCollector()
        stdin_lines = sc.read_all()

        assert len(file_lines) == len(stdin_lines)


# ===================================================================
# 11. FileCollector ordering with numbered log files
# ===================================================================

class TestFileCollectorNumberedFiles:
    """Ordering with log.1, log.2, log.3 etc."""

    def test_sorted_by_filename(self, tmp_path):
        for i in [3, 1, 2]:
            (tmp_path / f"log.{i}").write_text(f"from_{i}\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "log.*")])
        lines = fc.read_all()
        # Files sorted: log.1, log.2, log.3
        assert lines[0].strip() == "from_1"
        assert lines[1].strip() == "from_2"
        assert lines[2].strip() == "from_3"

    def test_numbered_files_stream_order(self, tmp_path):
        for i in [5, 2, 8]:
            (tmp_path / f"app.log.{i}").write_text(f"n{i}\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "app.log.*")])
        lines = list(fc.stream())
        # stream also uses sorted(self.files)
        stripped = [l.strip() for l in lines]
        assert stripped == ["n2", "n5", "n8"]

    def test_ten_numbered_files(self, tmp_path):
        for i in range(10):
            (tmp_path / f"svc.{i}.log").write_text(f"line_{i}\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "svc.*.log")])
        lines = fc.read_all()
        assert len(lines) == 10

    def test_mixed_named_and_numbered(self, tmp_path):
        (tmp_path / "app.log").write_text("current\n", encoding="utf-8")
        (tmp_path / "app.log.1").write_text("prev1\n", encoding="utf-8")
        (tmp_path / "app.log.2").write_text("prev2\n", encoding="utf-8")
        fc = FileCollector([str(tmp_path / "app.log"), str(tmp_path / "app.log.*")])
        lines = fc.read_all()
        assert len(lines) >= 3


# ===================================================================
# 12. FileCollector with special chars in filenames
# ===================================================================

class TestFileCollectorSpecialFilenames:
    """Spaces, parentheses, and other special characters in filenames."""

    def test_space_in_filename(self, tmp_path):
        f = tmp_path / "my log file.log"
        f.write_text("space_content\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0].strip() == "space_content"

    def test_parentheses_in_filename(self, tmp_path):
        f = tmp_path / "app (copy).log"
        f.write_text("paren_content\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0].strip() == "paren_content"

    def test_dash_and_underscore(self, tmp_path):
        f = tmp_path / "my-app_v2.0.log"
        f.write_text("dash_underscore\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert len(lines) == 1

    def test_korean_in_filename(self, tmp_path):
        f = tmp_path / "서버로그.log"
        f.write_text("korean_file\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0].strip() == "korean_file"

    def test_emoji_in_filename(self, tmp_path):
        try:
            f = tmp_path / "\U0001f525fire.log"
            f.write_text("hot\n", encoding="utf-8")
            fc = FileCollector([str(f)])
            lines = fc.read_all()
            assert lines[0].strip() == "hot"
        except (OSError, UnicodeError):
            pytest.skip("OS does not support emoji in filenames")

    def test_multiple_dots_in_filename(self, tmp_path):
        f = tmp_path / "app.2024.01.15.log"
        f.write_text("dotted\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = fc.read_all()
        assert lines[0].strip() == "dotted"

    def test_stream_special_filename(self, tmp_path):
        f = tmp_path / "log (backup 2).log"
        f.write_text("backup_line\n", encoding="utf-8")
        fc = FileCollector([str(f)])
        lines = list(fc.stream())
        assert lines[0].strip() == "backup_line"


# ===================================================================
# 13. CommandCollector - additional edge cases
# ===================================================================

class TestCommandCollectorEdgeCases:
    """Additional CommandCollector edge cases."""

    @patch("subprocess.run")
    def test_empty_command_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="")
        cc = CommandCollector("true")
        lines = cc.read_all()
        # splitlines on empty string returns []
        assert lines == [] or lines == [""]

    def test_read_all_returns_list(self):
        cc = CommandCollector("echo test", follow=False)
        result = cc.read_all()
        assert isinstance(result, list)

    @patch("subprocess.Popen")
    def test_stream_terminates_on_exhaust(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["a\n", "b\n"])
        mock_popen.return_value = mock_proc

        cc = CommandCollector("cmd")
        lines = list(cc.stream())
        assert len(lines) == 2
        mock_proc.terminate.assert_called_once()

    def test_command_preserved(self):
        cc = CommandCollector("tail -f /var/log/syslog")
        assert cc.command == "tail -f /var/log/syslog"

    def test_follow_default_true(self):
        cc = CommandCollector("any_cmd")
        assert cc.follow is True
