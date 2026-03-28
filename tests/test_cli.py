"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from logmind.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "learn" in result.output
        assert "scan" in result.output
        assert "watch" in result.output
        assert "label" in result.output
        assert "search" in result.output
        assert "stats" in result.output

    def test_learn_help(self, runner):
        result = runner.invoke(main, ["learn", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--window" in result.output

    def test_scan_help(self, runner):
        result = runner.invoke(main, ["scan", "--help"])
        assert result.exit_code == 0
        assert "--sensitivity" in result.output

    def test_watch_help(self, runner):
        result = runner.invoke(main, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--docker" in result.output
        assert "--slack" in result.output

    def test_label_help(self, runner):
        result = runner.invoke(main, ["label", "--help"])
        assert result.exit_code == 0
        assert "--label" in result.output
        assert "--resolution" in result.output

    def test_stats_no_index(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(main, ["stats"])
        assert result.exit_code != 0 or "No index" in result.output

    def test_search_no_index(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(main, ["search", "test"])
        assert "No index" in result.output or result.exit_code != 0

    def test_learn_missing_file(self, runner):
        result = runner.invoke(main, ["learn", "nonexistent.log"])
        assert result.exit_code != 0

    def test_scan_missing_index(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        log = tmp_path / "test.log"
        log.write_text("2024-01-01 10:00:00 INFO test")
        result = runner.invoke(main, ["scan", str(log)])
        assert "No index" in result.output or result.exit_code != 0
