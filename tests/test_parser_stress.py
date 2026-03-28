"""Comprehensive stress tests for logmind.parser module.

50+ test cases covering timestamp parsing, level detection, source extraction,
JSON log parsing, multiline logs, auto-format detection, and edge cases.
All tests are self-contained with no external file or model dependencies.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from logmind.parser import (
    _parse_level,
    _parse_source,
    _parse_timestamp,
    _try_json,
    auto_detect_format,
    parse_line,
    parse_lines,
)


# ============================================================
# 1. Timestamp parsing
# ============================================================

class TestTimestampParsing:
    """Tests for _parse_timestamp covering all format variants."""

    def test_iso8601_basic(self):
        ts = _parse_timestamp("2024-03-15T10:30:45Z")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 3
        assert ts.day == 15
        assert ts.hour == 10
        assert ts.minute == 30
        assert ts.second == 45

    def test_iso8601_fractional_seconds(self):
        ts = _parse_timestamp("2024-03-15T10:30:45.123Z")
        assert ts is not None
        assert ts.year == 2024

    def test_iso8601_no_z_suffix(self):
        ts = _parse_timestamp("2024-03-15T10:30:45")
        assert ts is not None
        assert ts.second == 45

    def test_iso8601_high_precision(self):
        ts = _parse_timestamp("2024-03-15T10:30:45.1234567Z")
        assert ts is not None

    def test_common_format_with_comma(self):
        ts = _parse_timestamp("2024-03-15 10:30:45,123")
        assert ts is not None
        assert ts.year == 2024
        assert ts.hour == 10

    def test_common_format_with_dot(self):
        ts = _parse_timestamp("2024-03-15 10:30:45.456")
        assert ts is not None

    def test_common_format_no_fractional(self):
        ts = _parse_timestamp("2024-03-15 10:30:45")
        assert ts is not None
        assert ts.second == 45

    def test_syslog_timestamp(self):
        ts = _parse_timestamp("Mar 15 10:30:45")
        assert ts is not None
        assert ts.month == 3
        assert ts.day == 15

    def test_syslog_single_digit_day(self):
        ts = _parse_timestamp("Jan  5 08:00:00")
        assert ts is not None
        assert ts.day == 5

    def test_nginx_timestamp(self):
        ts = _parse_timestamp("15/Mar/2024:10:30:45")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 3

    def test_epoch_seconds(self):
        ts = _parse_timestamp("1710500000.123")
        assert ts is not None
        assert ts.year >= 2024

    def test_epoch_integer(self):
        ts = _parse_timestamp("1710500000")
        assert ts is not None

    def test_no_timestamp_returns_none(self):
        assert _parse_timestamp("no timestamp here") is None

    def test_no_timestamp_random_text(self):
        assert _parse_timestamp("Hello world, just a message!") is None

    def test_timestamp_embedded_in_text(self):
        ts = _parse_timestamp("ERROR 2024-03-15 10:30:45,123 something broke")
        assert ts is not None
        assert ts.year == 2024

    def test_edge_midnight(self):
        ts = _parse_timestamp("2024-01-01T00:00:00Z")
        assert ts is not None
        assert ts.hour == 0
        assert ts.minute == 0

    def test_edge_end_of_day(self):
        ts = _parse_timestamp("2024-12-31T23:59:59Z")
        assert ts is not None
        assert ts.hour == 23
        assert ts.minute == 59


# ============================================================
# 2. Level detection
# ============================================================

class TestLevelDetection:
    """Tests for _parse_level with all levels, case variants, and HTTP codes."""

    def test_info_level(self):
        assert _parse_level("INFO Starting application") == "INFO"

    def test_error_level(self):
        assert _parse_level("ERROR Connection refused") == "ERROR"

    def test_warn_level(self):
        assert _parse_level("WARN Disk space low") == "WARN"

    def test_warning_level(self):
        assert _parse_level("WARNING deprecated API") == "WARN"

    def test_debug_level(self):
        assert _parse_level("DEBUG entering function") == "DEBUG"

    def test_fatal_level(self):
        assert _parse_level("FATAL system crash") == "FATAL"

    def test_critical_level(self):
        assert _parse_level("CRITICAL out of memory") == "FATAL"

    def test_crit_level(self):
        assert _parse_level("CRIT kernel panic") == "FATAL"

    def test_err_level(self):
        assert _parse_level("ERR failed to connect") == "ERROR"

    def test_trace_level(self):
        assert _parse_level("TRACE method entry") == "DEBUG"

    def test_verbose_level(self):
        assert _parse_level("VERBOSE full dump") == "DEBUG"

    def test_notice_level(self):
        assert _parse_level("NOTICE config reloaded") == "INFO"

    def test_case_insensitive_lowercase(self):
        assert _parse_level("error something bad") == "ERROR"

    def test_case_insensitive_mixed(self):
        assert _parse_level("Error connection lost") == "ERROR"

    def test_http_500_error(self):
        assert _parse_level('GET /api 500 0.123') == "ERROR"

    def test_http_503_error(self):
        assert _parse_level('POST /submit 503 0.5') == "ERROR"

    def test_http_404_warn(self):
        assert _parse_level('GET /missing 404 0.01') == "WARN"

    def test_http_400_warn(self):
        assert _parse_level('POST /bad 400 0.02') == "WARN"

    def test_no_level_defaults_info(self):
        assert _parse_level("just a regular message") == "INFO"

    def test_level_in_mixed_text(self):
        assert _parse_level("2024-01-01 10:00:00 ERROR db connection failed") == "ERROR"


# ============================================================
# 3. Source extraction
# ============================================================

class TestSourceExtraction:
    """Tests for _parse_source across different log formats."""

    def test_python_logger_format(self):
        line = "2024-03-15 10:30:45,123 - myapp.server - INFO - Starting"
        assert _parse_source(line) == "myapp.server"

    def test_python_logger_deep_module(self):
        line = "2024-03-15 10:30:45 - com.example.module.sub - DEBUG - test"
        assert _parse_source(line) == "com.example.module.sub"

    def test_simple_format_source(self):
        line = "2024-03-15 10:30:45 INFO MyService: request processed"
        assert _parse_source(line) == "MyService"

    def test_spring_format_source(self):
        line = "2024-03-15 10:30:45.123 INFO 1234 --- [main] c.m.app.MainClass : Starting"
        source = _parse_source(line)
        assert source == "MainClass"

    def test_syslog_source(self):
        line = "Mar 15 10:30:45 webserver01 nginx: access log"
        source = _parse_source(line)
        assert source == "nginx"

    def test_syslog_source_with_pid(self):
        line = "Mar 15 10:30:45 host01 sshd[12345]: accepted key"
        source = _parse_source(line)
        assert source == "sshd"

    def test_unknown_source_fallback(self):
        line = "Something happened without a clear source"
        assert _parse_source(line) == "unknown"

    def test_error_level_source(self):
        line = "2024-01-01 00:00:00 ERROR DatabasePool: connection timeout"
        assert _parse_source(line) == "DatabasePool"


# ============================================================
# 4. JSON log parsing
# ============================================================

class TestJsonLogParsing:
    """Tests for _try_json with various JSON structures."""

    def test_valid_json_all_fields(self):
        line = json.dumps({
            "timestamp": "2024-03-15T10:30:45Z",
            "level": "ERROR",
            "message": "Connection failed",
            "source": "db-pool",
        })
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.message == "Connection failed"
        assert entry.source == "db-pool"

    def test_json_with_time_key(self):
        line = json.dumps({"time": "2024-03-15 10:30:45", "msg": "hello"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.timestamp is not None
        assert entry.message == "hello"

    def test_json_with_ts_key(self):
        line = json.dumps({"ts": "2024-03-15T10:30:45Z", "msg": "ok"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.timestamp is not None

    def test_json_with_at_timestamp(self):
        line = json.dumps({"@timestamp": "2024-03-15T10:30:45Z", "message": "elk style"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.timestamp is not None

    def test_json_level_severity_key(self):
        line = json.dumps({"severity": "warn", "message": "caution"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "WARN"

    def test_json_level_loglevel_key(self):
        line = json.dumps({"loglevel": "debug", "text": "verbose stuff"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "DEBUG"
        assert entry.message == "verbose stuff"

    def test_json_source_logger_key(self):
        line = json.dumps({"logger": "com.app.Main", "message": "init"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "com.app.Main"

    def test_json_source_service_key(self):
        line = json.dumps({"service": "payment-api", "msg": "charge ok"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "payment-api"

    def test_json_source_component_key(self):
        line = json.dumps({"component": "auth", "message": "token expired"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "auth"

    def test_json_nested_metadata(self):
        data = {
            "timestamp": "2024-03-15T10:30:45Z",
            "level": "ERROR",
            "message": "fail",
            "source": "api",
            "request_id": "abc-123",
            "duration_ms": "450",
        }
        entry = _try_json(json.dumps(data))
        assert entry is not None
        assert "request_id" in entry.metadata
        assert entry.metadata["request_id"] == "abc-123"

    def test_json_missing_message_uses_raw(self):
        line = json.dumps({"level": "INFO", "source": "app"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.message == line.strip()

    def test_json_missing_level_defaults_info(self):
        line = json.dumps({"message": "no level"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "INFO"

    def test_json_missing_source_defaults_unknown(self):
        line = json.dumps({"message": "no source"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "unknown"

    def test_malformed_json_returns_none(self):
        assert _try_json("{not valid json}") is None

    def test_json_array_returns_none(self):
        assert _try_json("[1, 2, 3]") is None

    def test_json_string_returns_none(self):
        assert _try_json('"just a string"') is None

    def test_json_number_returns_none(self):
        assert _try_json("42") is None

    def test_empty_json_object(self):
        entry = _try_json("{}")
        assert entry is not None
        assert entry.message == "{}"


# ============================================================
# 5. Multiline logs
# ============================================================

class TestMultilineLogs:
    """Tests for parse_lines multiline support (stack traces, continuations)."""

    def test_java_stack_trace(self):
        lines = [
            "2024-03-15 10:30:45,123 ERROR java.lang.NullPointerException: null",
            "\tat com.app.Service.process(Service.java:42)",
            "\tat com.app.Controller.handle(Controller.java:18)",
            "\tat sun.reflect.NativeMethodAccessorImpl.invoke(Unknown Source)",
            "2024-03-15 10:30:46,000 INFO Recovery complete",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "NullPointerException" in entries[0].message
        assert "Service.java:42" in entries[0].message
        assert entries[1].level == "INFO"

    def test_python_traceback(self):
        lines = [
            "2024-03-15 10:30:45 ERROR Unhandled exception",
            "Traceback (most recent call last):",
            '  File "app.py", line 10, in <module>',
            '    raise ValueError("bad")',
            "ValueError: bad",
            "2024-03-15 10:30:46 INFO Next entry",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "Traceback" in entries[0].message
        assert 'ValueError: bad' in entries[0].message

    def test_continuation_lines_attached(self):
        lines = [
            "2024-03-15 10:30:45 INFO Request payload:",
            "    user=john action=login",
            "2024-03-15 10:30:46 INFO Done",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "john" in entries[0].message

    def test_multiline_no_continuation(self):
        lines = [
            "2024-03-15 10:30:45 INFO Line one",
            "2024-03-15 10:30:46 INFO Line two",
            "2024-03-15 10:30:47 INFO Line three",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 3

    def test_multiline_at_end_of_input(self):
        lines = [
            "2024-03-15 10:30:45 ERROR Crash",
            "  caused by: OOM",
            "  at allocator.c:99",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 1
        assert "OOM" in entries[0].message
        assert "allocator.c" in entries[0].message

    def test_empty_lines_skipped(self):
        lines = [
            "2024-03-15 10:30:45 INFO A",
            "",
            "   ",
            "2024-03-15 10:30:46 INFO B",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2


# ============================================================
# 6. Auto-format detection
# ============================================================

class TestAutoFormatDetection:
    """Tests for auto_detect_format with sufficient sample lines per format."""

    def test_detect_json(self):
        lines = [
            json.dumps({"timestamp": "2024-03-15T10:30:45Z", "level": "INFO", "message": f"msg {i}"})
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "json"

    def test_detect_python(self):
        lines = [
            f"2024-03-15 10:30:{i:02d},123 - myapp.module{i} - INFO - Message number {i}"
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "python"

    def test_detect_spring(self):
        lines = [
            f"2024-03-15 10:30:{i:02d}.123 INFO 1234 --- [main] c.m.app.Class{i} : Msg {i}"
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "spring"

    def test_detect_syslog(self):
        lines = [
            f"Mar 15 10:30:{i:02d} host01 service{i}: message {i}"
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "syslog"

    def test_detect_nginx_access(self):
        lines = [
            f'192.168.1.{i} - - [15/Mar/2024:10:30:{i:02d} +0000] "GET /path{i} HTTP/1.1" 200 {100 + i}'
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "nginx_access"

    def test_detect_docker(self):
        lines = [
            f"2024-03-15T10:30:{i:02d}.000Z stdout message number {i}"
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "docker"

    def test_detect_simple(self):
        lines = [
            f"2024-03-15 10:30:{i:02d},000 INFO Just a simple message {i}"
            for i in range(12)
        ]
        fmt = auto_detect_format(lines)
        # simple or python are both valid depending on match ordering;
        # the key is it returns something reasonable
        assert fmt in ("simple", "python")

    def test_detect_empty_returns_simple(self):
        assert auto_detect_format([]) == "simple"

    def test_detect_mixed_favors_majority(self):
        json_lines = [
            json.dumps({"timestamp": "2024-03-15T10:30:45Z", "message": f"j{i}"})
            for i in range(8)
        ]
        plain_lines = [
            f"2024-03-15 10:30:{i:02d} INFO plain {i}"
            for i in range(2)
        ]
        lines = json_lines + plain_lines
        assert auto_detect_format(lines) == "json"

    def test_detect_non_matching_returns_simple(self):
        lines = [
            "zzz no format here",
            "another random line",
            "still nothing",
            "nope",
            "not a log format",
            "hello world",
        ]
        assert auto_detect_format(lines) == "simple"


# ============================================================
# 7. Edge cases
# ============================================================

class TestEdgeCases:
    """Edge case tests: empty, unicode, long, binary, special chars."""

    def test_empty_line(self):
        entry = parse_line("")
        assert entry.message == ""
        assert entry.level == "DEBUG"
        assert entry.source == "unknown"
        assert entry.timestamp is None

    def test_whitespace_only_line(self):
        entry = parse_line("   \t  ")
        assert entry.message == ""
        assert entry.level == "DEBUG"

    def test_unicode_message(self):
        line = "2024-03-15 10:30:45 ERROR \uc11c\ubc84 \uc5f0\uacb0 \uc2e4\ud328: \ud0c0\uc784\uc544\uc6c3"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert "\uc11c\ubc84" in entry.message

    def test_unicode_emoji(self):
        line = "2024-03-15 10:30:45 WARN \u26a0\ufe0f disk usage high"
        entry = parse_line(line)
        assert entry.level == "WARN"

    def test_japanese_characters(self):
        line = "2024-03-15 10:30:45 ERROR \u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u63a5\u7d9a\u5931\u6557"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert entry.timestamp is not None

    def test_very_long_line(self):
        payload = "x" * 100_000
        line = f"2024-03-15 10:30:45 ERROR {payload}"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert len(entry.raw) > 100_000

    def test_binary_content_like(self):
        line = "2024-03-15 10:30:45 WARN \x00\x01\x02\x03\xff\xfe"
        entry = parse_line(line)
        assert entry.level == "WARN"
        assert entry.timestamp is not None

    def test_special_chars_brackets(self):
        line = "2024-03-15 10:30:45 INFO [main] <xml/> & 'quoted' \"double\""
        entry = parse_line(line)
        assert entry.level == "INFO"

    def test_newline_in_raw(self):
        entry = parse_line("2024-03-15 10:30:45 INFO hello\n")
        assert entry.raw == "2024-03-15 10:30:45 INFO hello"

    def test_carriage_return(self):
        entry = parse_line("2024-03-15 10:30:45 INFO hello\r\n")
        assert entry.raw == "2024-03-15 10:30:45 INFO hello"

    def test_tab_separated_fields(self):
        line = "2024-03-15 10:30:45\tERROR\ttimeout"
        entry = parse_line(line)
        assert entry.timestamp is not None

    def test_pipe_separated_fields(self):
        line = "2024-03-15 10:30:45 | ERROR | Something failed"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert entry.timestamp is not None

    def test_only_timestamp(self):
        line = "2024-03-15T10:30:45Z"
        entry = parse_line(line)
        assert entry.timestamp is not None

    def test_only_level(self):
        entry = parse_line("ERROR")
        assert entry.level == "ERROR"

    def test_repeated_levels_uses_first(self):
        entry = parse_line("2024-03-15 10:30:45 ERROR then WARN then DEBUG")
        assert entry.level == "ERROR"


# ============================================================
# 8. parse_line with explicit format
# ============================================================

class TestParseLineExplicitFormat:
    """Tests for parse_line when a specific format is given."""

    def test_python_format(self):
        line = "2024-03-15 10:30:45,123 - myapp - ERROR - disk full"
        entry = parse_line(line, fmt="python")
        assert entry.level == "ERROR"
        assert "disk full" in entry.message
        assert entry.source == "myapp"

    def test_syslog_format(self):
        line = "Mar 15 10:30:45 webhost nginx[123]: GET /index 200"
        entry = parse_line(line, fmt="syslog")
        assert entry.timestamp is not None

    def test_docker_format(self):
        line = "2024-03-15T10:30:45.000Z stdout container starting up"
        entry = parse_line(line, fmt="docker")
        assert entry.timestamp is not None
        assert "container starting up" in entry.message

    def test_json_format_explicit(self):
        line = json.dumps({"timestamp": "2024-03-15T10:30:45Z", "level": "WARN", "message": "slow query"})
        entry = parse_line(line, fmt="json")
        assert entry.level == "WARN"
        assert entry.message == "slow query"

    def test_unmatched_format_falls_back(self):
        line = "just plain text no format"
        entry = parse_line(line, fmt="python")
        # Falls through to generic parsing
        assert entry.message is not None
        assert entry.raw == line


# ============================================================
# 9. parse_lines integration
# ============================================================

class TestParseLinesIntegration:
    """Integration tests for parse_lines covering line numbers and mixed input."""

    def test_line_numbers_assigned(self):
        lines = [
            "2024-03-15 10:30:45 INFO First",
            "2024-03-15 10:30:46 INFO Second",
            "2024-03-15 10:30:47 INFO Third",
        ]
        entries = parse_lines(lines)
        assert entries[0].line_number == 1
        assert entries[1].line_number == 2
        assert entries[2].line_number == 3

    def test_mixed_json_and_plain(self):
        lines = [
            json.dumps({"timestamp": "2024-03-15T10:30:45Z", "level": "INFO", "message": "json line"}),
            "2024-03-15 10:30:46 ERROR plain line",
        ]
        entries = parse_lines(lines, fmt="auto")
        assert len(entries) == 2

    def test_single_line(self):
        entries = parse_lines(["2024-03-15 10:30:45 INFO single"])
        assert len(entries) == 1
        assert entries[0].message is not None

    def test_all_empty_lines(self):
        entries = parse_lines(["", "  ", "\t", "\n"])
        assert len(entries) == 0

    def test_large_batch(self):
        lines = [
            f"2024-03-15 10:{i // 60:02d}:{i % 60:02d} INFO msg-{i}"
            for i in range(500)
        ]
        entries = parse_lines(lines)
        assert len(entries) == 500

    def test_is_error_property(self):
        line = "2024-03-15 10:30:45 ERROR boom"
        entry = parse_line(line)
        assert entry.is_error is True
        assert entry.is_warn is False

    def test_is_warn_property(self):
        line = "2024-03-15 10:30:45 WARN caution"
        entry = parse_line(line)
        assert entry.is_warn is True
        assert entry.is_error is False

    def test_fatal_is_error(self):
        line = "2024-03-15 10:30:45 FATAL shutdown"
        entry = parse_line(line)
        assert entry.is_error is True


# ============================================================
# 10. JSON edge cases
# ============================================================

class TestJsonEdgeCases:
    """Additional JSON parsing edge cases."""

    def test_json_with_log_key(self):
        line = json.dumps({"log": "container output here"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.message == "container output here"

    def test_json_with_body_key(self):
        line = json.dumps({"body": "request body log"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.message == "request body log"

    def test_json_with_module_source(self):
        line = json.dumps({"module": "auth_handler", "message": "token check"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "auth_handler"

    def test_json_with_name_source(self):
        line = json.dumps({"name": "worker-1", "message": "processing"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.source == "worker-1"

    def test_json_datetime_key(self):
        line = json.dumps({"datetime": "2024-03-15T10:30:45Z", "message": "dt key"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.timestamp is not None

    def test_json_date_key(self):
        line = json.dumps({"date": "2024-03-15 10:30:45", "message": "date key"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.timestamp is not None

    def test_json_lvl_key(self):
        line = json.dumps({"lvl": "error", "message": "short key"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "ERROR"

    def test_json_log_level_key(self):
        line = json.dumps({"log_level": "fatal", "message": "crash"})
        entry = _try_json(line)
        assert entry is not None
        assert entry.level == "FATAL"

    def test_json_with_numeric_values(self):
        line = json.dumps({"timestamp": "2024-03-15T10:30:45Z", "level": "INFO", "message": "ok", "status": 200, "latency": 0.5})
        entry = _try_json(line)
        assert entry is not None
        assert entry.metadata["status"] == "200"
        assert entry.metadata["latency"] == "0.5"


# ============================================================
# 11. Regression / realistic logs
# ============================================================

class TestRealisticLogs:
    """Tests with real-world-style log lines."""

    def test_kubernetes_pod_log(self):
        line = '2024-03-15T10:30:45.123456Z stdout F {"level":"error","msg":"pod crash loop","pod":"web-abc123"}'
        entry = parse_line(line)
        assert entry.timestamp is not None

    def test_aws_cloudwatch_style(self):
        line = "2024-03-15T10:30:45.123Z\tRequestId: abc-123\tINFO\tLambda invoked"
        entry = parse_line(line)
        assert entry.timestamp is not None
        assert entry.level == "INFO"

    def test_apache_error_log_style(self):
        line = "[2024-03-15 10:30:45.123] ERROR proxy: ap_get_brigade failed"
        entry = parse_line(line)
        assert entry.level == "ERROR"

    def test_systemd_journal_style(self):
        line = "Mar 15 10:30:45 myhost systemd[1]: Started MyService."
        entry = parse_line(line)
        assert entry.timestamp is not None

    def test_multiline_json_stack_trace(self):
        lines = [
            json.dumps({
                "timestamp": "2024-03-15T10:30:45Z",
                "level": "ERROR",
                "message": "Unhandled exception",
                "source": "api-gw",
            }),
            "  java.lang.RuntimeException: timeout",
            "    at com.api.Handler.process(Handler.java:55)",
            json.dumps({
                "timestamp": "2024-03-15T10:30:46Z",
                "level": "INFO",
                "message": "Recovered",
                "source": "api-gw",
            }),
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "RuntimeException" in entries[0].message
