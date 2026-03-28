"""Tests for log parser."""

import pytest
from datetime import datetime
from logmind.parser import (
    parse_line, parse_lines, parse_file, auto_detect_format,
    _parse_timestamp, _parse_level, _parse_source,
)


class TestParseTimestamp:
    def test_iso8601(self):
        ts = _parse_timestamp("2024-03-15T10:30:45.123Z something")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 3
        assert ts.hour == 10

    def test_common_format(self):
        ts = _parse_timestamp("2024-03-15 10:30:45,123 INFO msg")
        assert ts is not None
        assert ts.minute == 30

    def test_syslog_format(self):
        ts = _parse_timestamp("Mar 15 10:30:45 server app: msg")
        assert ts is not None
        assert ts.day == 15

    def test_no_timestamp(self):
        ts = _parse_timestamp("just a plain message")
        assert ts is None


class TestParseLevel:
    def test_error(self):
        assert _parse_level("2024-01-01 ERROR something") == "ERROR"

    def test_warn(self):
        assert _parse_level("WARNING: disk almost full") == "WARN"

    def test_info(self):
        assert _parse_level("2024-01-01 INFO started") == "INFO"

    def test_fatal(self):
        assert _parse_level("FATAL: system crash") == "FATAL"

    def test_critical(self):
        assert _parse_level("CRITICAL error occurred") == "FATAL"

    def test_debug(self):
        assert _parse_level("DEBUG checking value") == "DEBUG"

    def test_http_500(self):
        assert _parse_level('GET /api 500 0.5s') == "ERROR"

    def test_http_404(self):
        assert _parse_level('GET /page 404 0.1s') == "WARN"

    def test_no_level(self):
        assert _parse_level("just some text") == "INFO"


class TestParseSource:
    def test_simple_format(self):
        src = _parse_source("2024-01-01 10:00:00 ERROR DBPool: connection failed")
        assert src == "DBPool"

    def test_python_logger(self):
        src = _parse_source("2024-01-01 10:00:00 - myapp.db - ERROR - failed")
        assert "myapp" in src or "db" in src.lower()

    def test_unknown(self):
        src = _parse_source("just text")
        assert src == "unknown"


class TestParseLine:
    def test_simple_log(self):
        entry = parse_line("2024-03-15 10:30:45 ERROR DBPool: Connection failed")
        assert entry.level == "ERROR"
        assert entry.timestamp is not None
        assert "DBPool" in entry.source or "Connection" in entry.message

    def test_json_log(self):
        line = '{"timestamp":"2024-03-15T10:30:45Z","level":"ERROR","message":"fail","source":"api"}'
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert entry.message == "fail"
        assert entry.source == "api"

    def test_empty_line(self):
        entry = parse_line("")
        assert entry.message == ""
        assert entry.level == "DEBUG"

    def test_plain_text(self):
        entry = parse_line("some random text")
        assert entry.raw == "some random text"


class TestParseLines:
    def test_multiline_stacktrace(self):
        lines = [
            "2024-03-15 10:30:45 ERROR App: NullPointerException",
            "  at com.app.Service.run(Service.java:42)",
            "  at com.app.Main.main(Main.java:10)",
            "2024-03-15 10:30:46 INFO App: Recovered",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "NullPointerException" in entries[0].message
        assert "Service.java:42" in entries[0].message

    def test_multiple_entries(self):
        lines = [
            "2024-03-15 10:00:00 INFO App: started",
            "2024-03-15 10:00:01 ERROR DB: failed",
            "2024-03-15 10:00:02 WARN Cache: miss",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 3
        assert entries[0].level == "INFO"
        assert entries[1].level == "ERROR"
        assert entries[2].level == "WARN"

    def test_empty_input(self):
        entries = parse_lines([])
        assert entries == []


class TestAutoDetectFormat:
    def test_json_format(self):
        lines = [
            '{"timestamp":"2024-01-01","level":"INFO","message":"ok"}',
            '{"timestamp":"2024-01-01","level":"ERROR","message":"fail"}',
        ]
        assert auto_detect_format(lines) == "json"

    def test_simple_format(self):
        lines = [
            "2024-03-15 10:00:00 INFO started",
            "2024-03-15 10:00:01 ERROR failed",
        ]
        fmt = auto_detect_format(lines)
        assert fmt in ("simple", "python", "syslog")  # any text format

    def test_empty(self):
        assert auto_detect_format([]) == "simple"


class TestParseFile:
    def test_sample_log(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text(
            "2024-01-01 10:00:00 INFO App: started\n"
            "2024-01-01 10:00:01 ERROR DB: connection failed\n"
            "2024-01-01 10:00:02 WARN Cache: eviction\n"
        )
        entries = parse_file(str(log))
        assert len(entries) == 3
        assert entries[1].is_error
