"""End-to-end CLI tests for LogMind using Click's CliRunner.

These tests exercise the full CLI pipeline: learn, stats, label, search, scan.
They use real sentence-transformers models and create temporary log files with
realistic timestamps.

Run with:  pytest tests/test_cli_e2e.py -v
Slow tests: pytest tests/test_cli_e2e.py -v -m slow
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

from logmind.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(base: datetime, offset_seconds: int) -> str:
    """Return a formatted timestamp string offset from *base*."""
    dt = base + timedelta(seconds=offset_seconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_log_file(
    path,
    n_lines: int = 80,
    error_rate: float = 0.1,
    window_seconds: int = 120,
    base_time: datetime | None = None,
):
    """Write a realistic log file with timestamps spaced >= 1 s apart."""
    base = base_time or datetime(2025, 6, 15, 10, 0, 0)
    lines = []
    for i in range(n_lines):
        ts = _ts(base, i * max(1, window_seconds // n_lines))
        if i % int(1 / error_rate) == 0 and error_rate > 0:
            level = "ERROR"
            msg = f"Connection timeout to db-server-{i % 3} after 30s"
        elif i % 7 == 0:
            level = "WARN"
            msg = f"Slow query detected: {i * 13 % 500}ms"
        else:
            level = "INFO"
            msg = f"Request processed successfully id={i}"
        lines.append(f"{ts} {level} AppServer: {msg}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_error_heavy_log(path, base_time: datetime | None = None, n_lines: int = 60):
    """Write a log dominated by ERROR lines (useful for scan anomaly detection)."""
    base = base_time or datetime(2025, 6, 15, 14, 0, 0)
    lines = []
    for i in range(n_lines):
        ts = _ts(base, i * 2)
        level = "ERROR" if i % 2 == 0 else "FATAL"
        msg = f"OutOfMemoryError: heap space exhausted attempt={i}"
        lines.append(f"{ts} {level} Worker: {msg}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_json_log(path, n_lines: int = 40):
    """Write a JSON-format log file."""
    base = datetime(2025, 6, 15, 10, 0, 0)
    lines = []
    for i in range(n_lines):
        ts = base + timedelta(seconds=i * 2)
        entry = {
            "timestamp": ts.isoformat(),
            "level": "ERROR" if i % 10 == 0 else "INFO",
            "source": "api-gateway",
            "message": f"Handled request {i}",
        }
        lines.append(json.dumps(entry) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """chdir into tmp_path so .logmind is created there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def log_file(workdir):
    """A standard log file in the working directory."""
    p = workdir / "app.log"
    _make_log_file(p)
    return p


@pytest.fixture()
def learned_index(runner, log_file, workdir):
    """Run 'learn' once and return the workdir that contains .logmind/."""
    result = runner.invoke(main, ["learn", str(log_file)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert (workdir / ".logmind" / "index.pkl").exists()
    return workdir


# ===================================================================
# 1. LEARN COMMAND
# ===================================================================

@pytest.mark.slow
class TestLearnCommand:
    """Tests for `logmind learn`."""

    def test_learn_basic(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file)])
        assert result.exit_code == 0
        assert "[OK] Index saved" in result.output
        assert "Windows:" in result.output

    def test_learn_creates_index_file(self, runner, log_file, workdir):
        runner.invoke(main, ["learn", str(log_file)])
        assert (workdir / ".logmind" / "index.pkl").exists()

    def test_learn_reports_line_count(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file)])
        assert "Loaded" in result.output
        assert "80" in result.output  # default n_lines

    def test_learn_with_window_option(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file), "--window", "30"])
        assert result.exit_code == 0
        assert "[OK]" in result.output

    def test_learn_with_format_option(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file), "--format", "simple"])
        assert result.exit_code == 0

    def test_learn_with_bits_option(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file), "--bits", "2"])
        assert result.exit_code == 0

    def test_learn_bits_out_of_range(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file), "--bits", "8"])
        assert result.exit_code != 0  # Click validation error

    def test_learn_missing_file(self, runner, workdir):
        result = runner.invoke(main, ["learn", "nonexistent.log"])
        assert result.exit_code != 0
        assert "ERROR" in (result.output + (result.stderr or ""))

    def test_learn_empty_file(self, runner, workdir):
        empty = workdir / "empty.log"
        empty.write_text("", encoding="utf-8")
        result = runner.invoke(main, ["learn", str(empty)])
        assert result.exit_code != 0
        assert "No log lines" in (result.output + (result.stderr or ""))

    def test_learn_multiple_files(self, runner, workdir):
        f1 = workdir / "app1.log"
        f2 = workdir / "app2.log"
        _make_log_file(f1, n_lines=40, base_time=datetime(2025, 6, 15, 10, 0, 0))
        _make_log_file(f2, n_lines=40, base_time=datetime(2025, 6, 15, 12, 0, 0))
        result = runner.invoke(main, ["learn", str(f1), str(f2)])
        assert result.exit_code == 0
        assert "[OK]" in result.output

    def test_learn_json_format(self, runner, workdir):
        jlog = workdir / "api.log"
        _make_json_log(jlog)
        result = runner.invoke(main, ["learn", str(jlog), "--format", "json"])
        assert result.exit_code == 0

    def test_learn_auto_detect_incidents(self, runner, workdir):
        log = workdir / "mixed.log"
        base = datetime(2025, 6, 15, 10, 0, 0)
        lines = []
        # Normal lines
        for i in range(50):
            ts = _ts(base, i * 2)
            lines.append(f"{ts} INFO AppServer: normal operation id={i}\n")
        # Spike of errors
        for i in range(50, 80):
            ts = _ts(base, i * 2)
            lines.append(f"{ts} ERROR AppServer: database connection refused\n")
        # More normal lines
        for i in range(80, 130):
            ts = _ts(base, i * 2)
            lines.append(f"{ts} INFO AppServer: normal operation id={i}\n")
        log.write_text("".join(lines), encoding="utf-8")
        result = runner.invoke(main, ["learn", str(log), "--auto-detect-incidents"])
        assert result.exit_code == 0

    def test_learn_glob_pattern(self, runner, workdir):
        for i in range(3):
            p = workdir / f"svc{i}.log"
            _make_log_file(p, n_lines=30, base_time=datetime(2025, 6, 15, 10 + i, 0, 0))
        result = runner.invoke(main, ["learn", str(workdir / "svc*.log")])
        assert result.exit_code == 0

    def test_learn_reports_sources(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file)])
        assert "Sources:" in result.output

    def test_learn_reports_errors_and_warnings(self, runner, log_file, workdir):
        result = runner.invoke(main, ["learn", str(log_file)])
        assert "Errors:" in result.output
        assert "Warnings:" in result.output


# ===================================================================
# 2. STATS COMMAND
# ===================================================================

@pytest.mark.slow
class TestStatsCommand:
    """Tests for `logmind stats`."""

    def test_stats_after_learn(self, runner, learned_index):
        result = runner.invoke(main, ["stats"])
        assert result.exit_code == 0
        assert "Index Statistics" in result.output

    def test_stats_shows_totals(self, runner, learned_index):
        result = runner.invoke(main, ["stats"])
        assert "Total log lines" in result.output
        assert "Total windows" in result.output

    def test_stats_shows_model_info(self, runner, learned_index):
        result = runner.invoke(main, ["stats"])
        assert "Embedding model" in result.output
        assert "MiniLM" in result.output

    def test_stats_no_index(self, runner, workdir):
        result = runner.invoke(main, ["stats"])
        assert result.exit_code != 0
        assert "No index found" in (result.output + (result.stderr or ""))


# ===================================================================
# 3. LABEL COMMAND
# ===================================================================

@pytest.mark.slow
class TestLabelCommand:
    """Tests for `logmind label`."""

    def test_label_basic(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "DB timeout incident",
        ])
        assert result.exit_code == 0
        assert "Incident labeled" in result.output

    def test_label_with_resolution(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "Memory leak",
            "--resolution", "Restarted pods and increased heap",
        ])
        assert result.exit_code == 0
        assert "Resolution:" in result.output

    def test_label_shows_total_incidents(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "Test incident",
        ])
        assert "Total incidents:" in result.output

    def test_label_invalid_time_format(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "not-a-date",
            "also-not-a-date",
            str(log_file),
            "--label", "Bad times",
        ])
        assert result.exit_code != 0
        assert "Cannot parse time" in (result.output + (result.stderr or ""))

    def test_label_no_entries_in_range(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2020-01-01 00:00",
            "2020-01-01 01:00",
            str(log_file),
            "--label", "Nothing here",
        ])
        assert result.exit_code != 0
        assert "No log entries found" in (result.output + (result.stderr or ""))

    def test_label_no_index(self, runner, workdir):
        log_file = workdir / "dummy.log"
        _make_log_file(log_file, n_lines=10)
        result = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "No index yet",
        ])
        assert result.exit_code != 0
        assert "No index found" in (result.output + (result.stderr or ""))

    def test_label_missing_required_label_option(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
        ])
        assert result.exit_code != 0  # --label is required

    def test_label_iso_time_format(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "label",
            "2025-06-15T10:00:00",
            "2025-06-15T10:01:00",
            str(log_file),
            "--label", "ISO format incident",
        ])
        assert result.exit_code == 0
        assert "Incident labeled" in result.output


# ===================================================================
# 4. SEARCH COMMAND
# ===================================================================

@pytest.mark.slow
class TestSearchCommand:
    """Tests for `logmind search`."""

    def test_search_no_incidents(self, runner, learned_index):
        result = runner.invoke(main, ["search", "timeout"])
        assert result.exit_code == 0
        assert "No incidents labeled" in result.output

    def test_search_after_label(self, runner, learned_index):
        log_file = learned_index / "app.log"
        # First label an incident
        runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "Database connection pool exhausted",
            "--resolution", "Increased pool size to 200",
        ])
        # Then search
        result = runner.invoke(main, ["search", "database connection"])
        assert result.exit_code == 0
        assert "Search Results" in result.output

    def test_search_with_k_option(self, runner, learned_index):
        log_file = learned_index / "app.log"
        runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:01",
            str(log_file),
            "--label", "OOM crash",
        ])
        result = runner.invoke(main, ["search", "memory", "-k", "1"])
        assert result.exit_code == 0

    def test_search_no_index(self, runner, workdir):
        result = runner.invoke(main, ["search", "anything"])
        assert result.exit_code != 0
        assert "No index found" in (result.output + (result.stderr or ""))

    def test_search_missing_query_argument(self, runner, learned_index):
        result = runner.invoke(main, ["search"])
        assert result.exit_code != 0  # missing required QUERY argument


# ===================================================================
# 5. SCAN COMMAND
# ===================================================================

@pytest.mark.slow
class TestScanCommand:
    """Tests for `logmind scan`."""

    def test_scan_basic(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, ["scan", str(log_file)])
        assert result.exit_code in (0, 1, 2)  # 0=ok, 1=warning, 2=critical
        assert "Scan Report" in result.output or "Scanning" in result.output

    def test_scan_json_output(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, ["scan", str(log_file), "--json"])
        assert result.exit_code in (0, 1, 2)
        # Should contain valid JSON
        data = json.loads(result.output)
        assert "total_windows" in data

    def test_scan_with_sensitivity_low(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, ["scan", str(log_file), "--sensitivity", "low"])
        assert result.exit_code in (0, 1, 2)

    def test_scan_with_sensitivity_high(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, ["scan", str(log_file), "--sensitivity", "high"])
        assert result.exit_code in (0, 1, 2)

    def test_scan_with_window_option(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, ["scan", str(log_file), "--window", "30"])
        assert result.exit_code in (0, 1, 2)

    def test_scan_error_heavy_file(self, runner, learned_index):
        error_log = learned_index / "errors.log"
        _make_error_heavy_log(error_log)
        result = runner.invoke(main, ["scan", str(error_log)])
        assert result.exit_code in (0, 1, 2)

    def test_scan_no_index(self, runner, workdir):
        log_file = workdir / "app.log"
        _make_log_file(log_file, n_lines=20)
        result = runner.invoke(main, ["scan", str(log_file)])
        assert result.exit_code != 0
        assert "No index found" in (result.output + (result.stderr or ""))

    def test_scan_missing_file(self, runner, learned_index):
        result = runner.invoke(main, ["scan", "ghost.log"])
        assert result.exit_code != 0
        assert "ERROR" in (result.output + (result.stderr or ""))

    def test_scan_invalid_sensitivity(self, runner, learned_index):
        log_file = learned_index / "app.log"
        result = runner.invoke(main, [
            "scan", str(log_file), "--sensitivity", "ultra"
        ])
        assert result.exit_code != 0  # Click choice validation

    def test_scan_multiple_files(self, runner, learned_index):
        f1 = learned_index / "app.log"
        f2 = learned_index / "app2.log"
        _make_log_file(f2, n_lines=30, base_time=datetime(2025, 6, 15, 12, 0, 0))
        result = runner.invoke(main, ["scan", str(f1), str(f2)])
        assert result.exit_code in (0, 1, 2)


# ===================================================================
# 6. VERSION AND HELP
# ===================================================================

class TestVersionAndHelp:
    """Non-slow tests for CLI surface area."""

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "logmind" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "learn" in result.output
        assert "scan" in result.output
        assert "watch" in result.output

    def test_learn_help(self, runner):
        result = runner.invoke(main, ["learn", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--window" in result.output

    def test_scan_help(self, runner):
        result = runner.invoke(main, ["scan", "--help"])
        assert result.exit_code == 0
        assert "--sensitivity" in result.output
        assert "--json" in result.output

    def test_label_help(self, runner):
        result = runner.invoke(main, ["label", "--help"])
        assert result.exit_code == 0
        assert "--label" in result.output

    def test_search_help(self, runner):
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "-k" in result.output

    def test_stats_help(self, runner):
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0

    def test_watch_help(self, runner):
        result = runner.invoke(main, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--docker" in result.output
        assert "--slack" in result.output

    def test_unknown_command(self, runner):
        result = runner.invoke(main, ["frobnicate"])
        assert result.exit_code != 0


# ===================================================================
# 7. FULL PIPELINE (learn -> label -> search -> stats -> scan)
# ===================================================================

@pytest.mark.slow
class TestFullPipeline:
    """Integration tests exercising a full workflow."""

    def test_full_workflow(self, runner, workdir):
        # 1. Create log file
        log_file = workdir / "pipeline.log"
        _make_log_file(log_file, n_lines=100, window_seconds=300)

        # 2. Learn
        res = runner.invoke(main, ["learn", str(log_file)])
        assert res.exit_code == 0, res.output + (res.stderr or "")
        assert "[OK]" in res.output

        # 3. Stats (before labeling)
        res = runner.invoke(main, ["stats"])
        assert res.exit_code == 0
        assert "Labeled incidents:    0" in res.output

        # 4. Label an incident
        res = runner.invoke(main, [
            "label",
            "2025-06-15 10:00",
            "2025-06-15 10:02",
            str(log_file),
            "--label", "Connection pool exhausted",
            "--resolution", "Increased max connections",
        ])
        assert res.exit_code == 0
        assert "Incident labeled" in res.output

        # 5. Stats (after labeling)
        res = runner.invoke(main, ["stats"])
        assert res.exit_code == 0
        assert "Labeled incidents:    1" in res.output

        # 6. Search for similar incidents
        res = runner.invoke(main, ["search", "connection pool"])
        assert res.exit_code == 0
        assert "Search Results" in res.output

        # 7. Scan the same file for anomalies
        res = runner.invoke(main, ["scan", str(log_file)])
        assert res.exit_code in (0, 1, 2)

    def test_learn_twice_overwrites_index(self, runner, workdir):
        log1 = workdir / "first.log"
        log2 = workdir / "second.log"
        _make_log_file(log1, n_lines=40, base_time=datetime(2025, 6, 15, 8, 0, 0))
        _make_log_file(log2, n_lines=60, base_time=datetime(2025, 6, 15, 12, 0, 0))

        # Learn from first file
        res1 = runner.invoke(main, ["learn", str(log1)])
        assert res1.exit_code == 0

        # Learn again from second file (overwrites)
        res2 = runner.invoke(main, ["learn", str(log2)])
        assert res2.exit_code == 0

        # Stats should reflect second learn
        res3 = runner.invoke(main, ["stats"])
        assert res3.exit_code == 0
        assert "60" in res3.output  # 60 lines from second file
