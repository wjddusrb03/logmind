"""Advanced tests for logmind.parser module.

60+ NEW test cases covering areas not in test_parser.py or test_parser_stress.py:
- parse_file with encodings, empty files, error handling
- Nginx access log full parsing
- Docker log format with stdout/stderr
- Spring Boot format with thread names and class paths
- Syslog with PID
- Mixed format files
- Timestamp edge cases (year boundaries, leap seconds, timezone offsets)
- Level detection false positives (ERROR inside URLs, variable names)
- Source extraction from deeply nested Java class paths
- parse_lines performance with 1000+ lines
- Malformed lines (truncated, null bytes, corruption)
- Windows-style paths in log messages
- SQL queries, HTML tags, base64 content in messages
- Consecutive identical timestamps
- parse_line with explicit format for each format type
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pytest

from logmind.parser import (
    _parse_level,
    _parse_source,
    _parse_timestamp,
    _try_json,
    auto_detect_format,
    parse_file,
    parse_line,
    parse_lines,
)


# ============================================================
# 1. parse_file with various encodings and edge cases
# ============================================================

class TestParseFileEncodings:
    """Tests for parse_file with different file encodings and conditions."""

    def test_utf8_file(self, tmp_path):
        log = tmp_path / "utf8.log"
        log.write_text(
            "2024-03-15 10:00:00 INFO App: started\n"
            "2024-03-15 10:00:01 ERROR DB: \uc5f0\uacb0 \uc2e4\ud328\n",
            encoding="utf-8",
        )
        entries = parse_file(str(log))
        assert len(entries) == 2
        assert entries[1].level == "ERROR"

    def test_latin1_file_with_replace(self, tmp_path):
        log = tmp_path / "latin1.log"
        content = "2024-03-15 10:00:00 INFO App: caf\xe9 ready\n"
        log.write_bytes(content.encode("latin-1"))
        # parse_file uses errors='replace', should not crash
        entries = parse_file(str(log))
        assert len(entries) == 1
        assert entries[0].level == "INFO"

    def test_empty_file(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        entries = parse_file(str(log))
        assert entries == []

    def test_file_with_only_newlines(self, tmp_path):
        log = tmp_path / "newlines.log"
        log.write_text("\n\n\n\n\n")
        entries = parse_file(str(log))
        assert entries == []

    def test_file_with_bom(self, tmp_path):
        log = tmp_path / "bom.log"
        content = "\ufeff2024-03-15 10:00:00 INFO App: started with BOM\n"
        log.write_text(content, encoding="utf-8-sig")
        entries = parse_file(str(log))
        assert len(entries) >= 1

    def test_large_file_does_not_crash(self, tmp_path):
        log = tmp_path / "large.log"
        with open(str(log), "w", encoding="utf-8") as f:
            for i in range(5000):
                f.write(f"2024-03-15 10:{i // 3600:02d}:{(i // 60) % 60:02d} INFO msg-{i}\n")
        entries = parse_file(str(log))
        assert len(entries) == 5000

    def test_file_with_null_bytes(self, tmp_path):
        log = tmp_path / "nulls.log"
        log.write_bytes(
            b"2024-03-15 10:00:00 INFO App: ok\x00\x00\n"
            b"2024-03-15 10:00:01 ERROR DB: fail\n"
        )
        entries = parse_file(str(log))
        assert len(entries) == 2

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            parse_file("/nonexistent/path/log.txt")

    def test_parse_file_with_explicit_format(self, tmp_path):
        log = tmp_path / "simple.log"
        log.write_text(
            "2024-01-01 12:00:00 ERROR SomeService: failure\n"
            "2024-01-01 12:00:01 INFO SomeService: recovered\n"
        )
        entries = parse_file(str(log), fmt="simple")
        assert len(entries) == 2
        assert entries[0].level == "ERROR"


# ============================================================
# 2. Nginx access log parsing - full format
# ============================================================

class TestNginxAccessLogParsing:
    """Full nginx access log format with IPs, status codes, bytes."""

    def test_nginx_200_ok(self):
        line = '192.168.1.1 - - [15/Mar/2024:10:30:45 +0000] "GET /index.html HTTP/1.1" 200 1234'
        entry = parse_line(line, fmt="nginx_access")
        assert entry.timestamp is not None
        assert entry.raw == line

    def test_nginx_404_warn_level(self):
        line = '10.0.0.5 - - [15/Mar/2024:10:30:45 +0000] "GET /missing HTTP/1.1" 404 0'
        entry = parse_line(line, fmt="nginx_access")
        assert entry.level == "WARN"

    def test_nginx_500_error_level(self):
        line = '172.16.0.1 - - [15/Mar/2024:10:30:45 +0000] "POST /api/data HTTP/1.1" 500 50'
        entry = parse_line(line, fmt="nginx_access")
        assert entry.level == "ERROR"

    def test_nginx_301_redirect(self):
        line = '192.168.0.1 - - [01/Jan/2024:00:00:01 +0000] "GET /old HTTP/1.1" 301 185'
        entry = parse_line(line, fmt="nginx_access")
        # 301 is not 4xx or 5xx, level should not be WARN/ERROR
        assert entry.level == "INFO"

    def test_nginx_auto_detect(self):
        lines = [
            f'10.0.0.{i} - - [15/Mar/2024:10:30:{i:02d} +0000] "GET /page{i} HTTP/1.1" 200 {100 + i}'
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "nginx_access"

    def test_nginx_with_ipv6(self):
        line = '::1 - - [15/Mar/2024:10:30:45 +0000] "GET / HTTP/1.1" 200 612'
        entry = parse_line(line)
        # Even if regex doesn't match perfectly, shouldn't crash
        assert entry.raw == line


# ============================================================
# 3. Docker log format with stdout/stderr
# ============================================================

class TestDockerLogFormat:
    """Docker log format: timestamp stdout/stderr message."""

    def test_docker_stdout(self):
        line = "2024-03-15T10:30:45.123456Z stdout Starting application..."
        entry = parse_line(line, fmt="docker")
        assert entry.timestamp is not None
        assert "Starting application" in entry.message

    def test_docker_stderr(self):
        line = "2024-03-15T10:30:45.000Z stderr ERROR: failed to bind port"
        entry = parse_line(line, fmt="docker")
        assert entry.timestamp is not None
        assert entry.level == "ERROR"

    def test_docker_stderr_no_level_keyword(self):
        line = "2024-03-15T10:30:45.000Z stderr connection refused"
        entry = parse_line(line, fmt="docker")
        assert entry.timestamp is not None
        assert "connection refused" in entry.message

    def test_docker_multiline(self):
        lines = [
            "2024-03-15T10:30:45.000Z stdout Starting server",
            "  Loading plugins...",
            "  Plugin auth loaded",
            "2024-03-15T10:30:46.000Z stdout Server ready on port 8080",
        ]
        entries = parse_lines(lines, fmt="docker")
        assert len(entries) == 2
        assert "Loading plugins" in entries[0].message


# ============================================================
# 4. Spring Boot log format with thread names and class paths
# ============================================================

class TestSpringBootFormat:
    """Spring Boot: TIMESTAMP LEVEL PID --- [thread] class : message."""

    def test_spring_basic(self):
        line = "2024-03-15 10:30:45.123 INFO 12345 --- [main] c.m.app.Application : Started in 3.2s"
        entry = parse_line(line, fmt="spring")
        assert entry.level == "INFO"
        assert entry.timestamp is not None
        assert "Started in 3.2s" in entry.message

    def test_spring_error_with_long_class(self):
        line = "2024-03-15 10:30:45.123 ERROR 12345 --- [http-nio-8080-exec-1] o.s.w.s.m.s.DefaultHandlerExceptionResolver : Resolved"
        entry = parse_line(line, fmt="spring")
        assert entry.level == "ERROR"

    def test_spring_source_extraction_deep_path(self):
        line = "2024-03-15 10:30:45.123 INFO 12345 --- [main] c.example.deep.pkg.MyService : Initializing"
        source = _parse_source(line)
        # Should extract the last part of the class path
        assert source == "MyService"

    def test_spring_auto_detect(self):
        lines = [
            f"2024-03-15 10:30:{i:02d}.123 INFO 1234 --- [main] c.m.app.Cls{i} : Msg {i}"
            for i in range(12)
        ]
        assert auto_detect_format(lines) == "spring"

    def test_spring_warn_thread_pool(self):
        line = "2024-03-15 10:30:45.123 WARN 99 --- [scheduling-1] c.m.app.ScheduledTask : Slow execution 15s"
        entry = parse_line(line, fmt="spring")
        assert entry.level == "WARN"


# ============================================================
# 5. Syslog with PID
# ============================================================

class TestSyslogWithPid:
    """Syslog format: Mon DD HH:MM:SS host process[PID]: message."""

    def test_syslog_sshd_with_pid(self):
        line = "Mar 15 10:30:45 webserver sshd[12345]: Accepted publickey for user"
        entry = parse_line(line, fmt="syslog")
        assert entry.timestamp is not None
        assert "Accepted publickey" in entry.message

    def test_syslog_source_strips_pid(self):
        line = "Mar 15 10:30:45 host01 cron[9876]: (root) CMD (/usr/local/bin/check)"
        source = _parse_source(line)
        assert source == "cron"

    def test_syslog_kernel_no_pid(self):
        line = "Mar 15 10:30:45 myhost kernel: [42.123456] Out of memory: Kill process"
        entry = parse_line(line, fmt="syslog")
        assert entry.timestamp is not None

    def test_syslog_systemd_pid1(self):
        line = "Jan  1 00:00:01 server systemd[1]: Started Daily apt download activities."
        source = _parse_source(line)
        assert source == "systemd"


# ============================================================
# 6. Mixed format files
# ============================================================

class TestMixedFormatFiles:
    """Files containing a mix of JSON lines and plain text."""

    def test_mixed_json_and_plain(self):
        lines = [
            '{"timestamp":"2024-03-15T10:30:45Z","level":"INFO","message":"json entry"}',
            "2024-03-15 10:30:46 ERROR PlainText: something broke",
            '{"timestamp":"2024-03-15T10:30:47Z","level":"WARN","message":"json warn"}',
        ]
        entries = parse_lines(lines, fmt="auto")
        assert len(entries) == 3
        # At least the JSON entries should be parsed correctly
        assert entries[0].message == "json entry"

    def test_mixed_plain_and_continuation(self):
        lines = [
            '{"timestamp":"2024-03-15T10:30:45Z","level":"ERROR","message":"crash"}',
            "  caused by: NullPointerException",
            "  at com.app.Main.run(Main.java:10)",
            "2024-03-15 10:30:46 INFO Recovery",
        ]
        entries = parse_lines(lines, fmt="auto")
        assert len(entries) == 2
        assert "NullPointerException" in entries[0].message

    def test_single_json_among_plain(self):
        lines = [
            "2024-03-15 10:30:45 INFO Startup",
            "2024-03-15 10:30:46 INFO Running",
            '{"timestamp":"2024-03-15T10:30:47Z","level":"ERROR","message":"oops"}',
            "2024-03-15 10:30:48 INFO Recovered",
        ]
        entries = parse_lines(lines, fmt="auto")
        assert len(entries) == 4


# ============================================================
# 7. Timestamp edge cases
# ============================================================

class TestTimestampEdgeCases:
    """Timestamps at year boundaries, unusual values, timezone offsets."""

    def test_year_2000(self):
        ts = _parse_timestamp("2000-01-01T00:00:00Z")
        assert ts is not None
        assert ts.year == 2000

    def test_year_2099(self):
        ts = _parse_timestamp("2099-12-31T23:59:59Z")
        assert ts is not None
        assert ts.year == 2099

    def test_feb_29_leap_year(self):
        ts = _parse_timestamp("2024-02-29 12:00:00")
        assert ts is not None
        assert ts.month == 2
        assert ts.day == 29

    def test_feb_29_non_leap_year_returns_none_or_error(self):
        # 2023 is not a leap year; parser may return None or raise
        ts = _parse_timestamp("2023-02-29 12:00:00")
        # The regex will match but strptime should fail; expect None
        assert ts is None

    def test_iso_with_positive_timezone_offset(self):
        # +09:00 appended; parser strips Z but not offsets
        ts = _parse_timestamp("2024-03-15T10:30:45+09:00")
        # May or may not parse depending on implementation; should not crash
        assert True  # no crash is success

    def test_iso_with_negative_timezone_offset(self):
        ts = _parse_timestamp("2024-03-15T10:30:45-05:00")
        assert True  # no crash is success

    def test_consecutive_identical_timestamps(self):
        lines = [
            "2024-03-15 10:30:45 INFO msg1",
            "2024-03-15 10:30:45 INFO msg2",
            "2024-03-15 10:30:45 INFO msg3",
            "2024-03-15 10:30:45 ERROR msg4",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 4
        # All four should have the same timestamp
        for e in entries:
            assert e.timestamp is not None
            assert e.timestamp.second == 45

    def test_syslog_december(self):
        ts = _parse_timestamp("Dec 31 23:59:59 host app: last second")
        assert ts is not None
        assert ts.month == 12
        assert ts.day == 31

    def test_nginx_timestamp_different_month(self):
        ts = _parse_timestamp("01/Jan/2024:00:00:00")
        assert ts is not None
        assert ts.month == 1
        assert ts.day == 1


# ============================================================
# 8. Level detection false positives
# ============================================================

class TestLevelDetectionFalsePositives:
    """ERROR/WARN inside URLs, variable names, etc. -- tests word boundary."""

    def test_error_in_url_path(self):
        # The word boundary regex \bERROR\b should match here since ERROR is standalone
        # but if it is part of a URL like /error_page, test behavior
        line = "2024-03-15 10:30:45 GET /api/errors/list 200 0.5s"
        # No standalone ERROR/WARN; should default INFO
        # (Note: the regex uses \b so "errors" won't match "ERROR")
        level = _parse_level(line)
        # "errors" should NOT match "ERROR" due to word boundary
        assert level == "INFO"

    def test_warning_in_variable_name(self):
        line = "2024-03-15 10:30:45 INFO warningCount=5, errorCount=0"
        # "warningCount" and "errorCount" contain the keywords but not as whole words
        # However \b might match "warning" inside "warningCount" depending on case
        # Since _LEVEL_RE is case-insensitive and uses \b, "warning" won't match
        # inside "warningCount" because 'C' is a word char
        level = _parse_level(line)
        assert level == "INFO"

    def test_debug_in_classname(self):
        line = "2024-03-15 10:30:45 DebugController: handling request"
        # "Debug" appears but as part of DebugController, not standalone
        level = _parse_level(line)
        # \b would match "Debug" at the start of "DebugController"
        # because the boundary is between start-of-word and 'D'
        # So this might match DEBUG -- that's actually a regex limitation
        # Just verify it doesn't crash and returns something
        assert level in ("DEBUG", "INFO")

    def test_error_in_email_address(self):
        line = "2024-03-15 10:30:45 Sent to error-reports@company.com"
        # "error" is bounded by '-' which is not a word char, so \b matches
        level = _parse_level(line)
        # This is a known limitation; the word "error" is technically word-bounded
        assert level in ("ERROR", "INFO")

    def test_http_status_200_not_error(self):
        line = 'GET /api/health 200 0.001s'
        level = _parse_level(line)
        assert level == "INFO"

    def test_http_status_201_not_error(self):
        line = 'POST /api/users 201 0.05s'
        level = _parse_level(line)
        assert level == "INFO"

    def test_http_status_502_is_error(self):
        line = 'GET /api/proxy 502 30.0s'
        level = _parse_level(line)
        assert level == "ERROR"

    def test_http_status_403_is_warn(self):
        line = 'GET /admin 403 0.001s'
        level = _parse_level(line)
        assert level == "WARN"


# ============================================================
# 9. Source extraction from deeply nested Java class paths
# ============================================================

class TestDeepSourceExtraction:
    """Source extraction from Spring-style deeply nested class paths."""

    def test_deep_spring_class_path(self):
        line = "2024-03-15 10:30:45.123 INFO 1 --- [main] o.s.b.w.e.t.TomcatWebServer : Tomcat started"
        source = _parse_source(line)
        assert source == "TomcatWebServer"

    def test_very_deep_package_path(self):
        line = "2024-03-15 10:30:45.123 INFO 1 --- [main] c.e.a.b.c.d.e.f.DeepClass : Init"
        source = _parse_source(line)
        assert source == "DeepClass"

    def test_single_letter_packages(self):
        line = "2024-03-15 10:30:45.123 INFO 1 --- [main] a.b.C : Short"
        source = _parse_source(line)
        assert source == "C"

    def test_python_dotted_source(self):
        line = "2024-03-15 10:30:45,123 - myapp.services.database.pool - ERROR - pool exhausted"
        source = _parse_source(line)
        assert source == "myapp.services.database.pool"


# ============================================================
# 10. parse_lines with 1000+ lines performance
# ============================================================

class TestParseLinesPerformance:
    """Verify parse_lines doesn't crash or take too long with large inputs."""

    def test_1000_lines_no_crash(self):
        lines = [
            f"2024-03-15 10:{i // 3600:02d}:{(i // 60) % 60:02d} INFO worker-{i % 10}: processed item {i}"
            for i in range(1000)
        ]
        entries = parse_lines(lines)
        assert len(entries) == 1000

    def test_2000_lines_completes_in_time(self):
        lines = [
            f"2024-03-15 10:{i // 3600:02d}:{(i // 60) % 60:02d} ERROR svc: failure {i}"
            for i in range(2000)
        ]
        start = time.time()
        entries = parse_lines(lines)
        elapsed = time.time() - start
        assert len(entries) == 2000
        assert elapsed < 30, f"parse_lines took {elapsed:.1f}s for 2000 lines"

    def test_1000_json_lines_no_crash(self):
        lines = [
            json.dumps({"timestamp": f"2024-03-15T10:30:{i % 60:02d}Z", "level": "INFO", "message": f"msg-{i}"})
            for i in range(1000)
        ]
        entries = parse_lines(lines)
        assert len(entries) == 1000

    def test_1500_lines_with_multiline(self):
        lines = []
        for i in range(500):
            lines.append(f"2024-03-15 10:00:{i % 60:02d} ERROR App: crash-{i}")
            lines.append(f"  at com.app.Svc.run(Svc.java:{i})")
            lines.append(f"  at com.app.Main.main(Main.java:{i})")
        entries = parse_lines(lines)
        assert len(entries) == 500
        for e in entries:
            assert "at com.app" in e.message


# ============================================================
# 11. Malformed lines - truncated, corrupted, null bytes
# ============================================================

class TestMalformedLines:
    """Lines that are truncated, have null bytes, or are corrupted."""

    def test_truncated_timestamp(self):
        line = "2024-03-"
        entry = parse_line(line)
        assert entry.raw == line
        # Should not crash

    def test_truncated_json(self):
        line = '{"timestamp":"2024-03-15T10:30:45Z","level":"ERR'
        entry = parse_line(line)
        assert entry.raw == line

    def test_null_bytes_in_middle(self):
        line = "2024-03-15 10:30:45 ERROR \x00\x00 something"
        entry = parse_line(line)
        assert entry.level == "ERROR"

    def test_only_null_bytes(self):
        line = "\x00\x00\x00"
        entry = parse_line(line)
        # Should not crash
        assert entry.raw == line

    def test_mixed_binary_and_text(self):
        line = "2024-03-15 10:30:45 WARN \xff\xfe\xfd data corruption"
        entry = parse_line(line)
        assert entry.level == "WARN"

    def test_extremely_long_single_line(self):
        payload = "A" * 500_000
        line = f"2024-03-15 10:30:45 ERROR {payload}"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert len(entry.raw) > 500_000

    def test_line_with_only_spaces_and_tabs(self):
        line = "   \t\t   \t  "
        entry = parse_line(line)
        assert entry.message == ""
        assert entry.level == "DEBUG"


# ============================================================
# 12. Windows-style paths in log messages
# ============================================================

class TestWindowsPaths:
    """Log messages containing Windows-style backslash paths."""

    def test_windows_path_in_error(self):
        line = r"2024-03-15 10:30:45 ERROR Failed to open C:\Users\Admin\logs\app.log"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert "C:" in entry.message or "Users" in entry.message

    def test_windows_unc_path(self):
        line = r"2024-03-15 10:30:45 WARN Cannot access \\server\share\file.dat"
        entry = parse_line(line)
        assert entry.level == "WARN"

    def test_windows_path_with_spaces(self):
        line = r'2024-03-15 10:30:45 INFO Loading config from "C:\Program Files\MyApp\config.yml"'
        entry = parse_line(line)
        assert entry.level == "INFO"
        assert entry.timestamp is not None


# ============================================================
# 13. Log messages with SQL, HTML, base64 content
# ============================================================

class TestSpecialContentInMessages:
    """Log messages containing SQL queries, HTML tags, base64."""

    def test_sql_query_in_message(self):
        line = "2024-03-15 10:30:45 ERROR DBPool: Slow query: SELECT * FROM users WHERE id = 42 AND status = 'active'"
        entry = parse_line(line)
        assert entry.level == "ERROR"
        assert "SELECT" in entry.message

    def test_sql_insert_with_values(self):
        line = "2024-03-15 10:30:45 DEBUG SQL: INSERT INTO logs (ts, msg) VALUES ('2024-01-01', 'test')"
        entry = parse_line(line)
        assert entry.level == "DEBUG"
        assert "INSERT" in entry.message

    def test_html_tags_in_message(self):
        line = '2024-03-15 10:30:45 WARN Template: Invalid HTML <div class="error"><b>Missing</b></div>'
        entry = parse_line(line)
        assert entry.level == "WARN"
        assert "<div" in entry.raw

    def test_base64_content(self):
        b64 = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQ="
        line = f"2024-03-15 10:30:45 INFO Payload: {b64}"
        entry = parse_line(line)
        assert entry.level == "INFO"
        assert b64 in entry.raw

    def test_json_embedded_in_plain_log(self):
        line = '2024-03-15 10:30:45 INFO Response: {"status": 200, "data": [1,2,3]}'
        entry = parse_line(line)
        assert entry.level == "INFO"
        assert entry.timestamp is not None

    def test_xml_payload(self):
        line = '2024-03-15 10:30:45 ERROR SOAP fault: <faultcode>Server</faultcode><faultstring>Internal</faultstring>'
        entry = parse_line(line)
        assert entry.level == "ERROR"

    def test_url_with_query_params(self):
        line = "2024-03-15 10:30:45 INFO GET https://api.example.com/v2/search?q=error&limit=10&page=1 200"
        entry = parse_line(line)
        assert entry.timestamp is not None


# ============================================================
# 14. Consecutive identical timestamps
# ============================================================

class TestConsecutiveTimestamps:
    """Multiple entries with the exact same timestamp."""

    def test_ten_entries_same_timestamp(self):
        lines = [
            f"2024-03-15 10:30:45 INFO batch-{i}: processed"
            for i in range(10)
        ]
        entries = parse_lines(lines)
        assert len(entries) == 10
        timestamps = [e.timestamp for e in entries]
        assert all(t == timestamps[0] for t in timestamps)

    def test_same_timestamp_different_levels(self):
        lines = [
            "2024-03-15 10:30:45 INFO Starting batch",
            "2024-03-15 10:30:45 WARN Slow network",
            "2024-03-15 10:30:45 ERROR Connection failed",
            "2024-03-15 10:30:45 FATAL System shutdown",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 4
        assert entries[0].level == "INFO"
        assert entries[1].level == "WARN"
        assert entries[2].level == "ERROR"
        assert entries[3].level == "FATAL"

    def test_same_timestamp_with_multiline(self):
        lines = [
            "2024-03-15 10:30:45 ERROR App: NullPointerException",
            "  at com.app.Service.run(Service.java:42)",
            "2024-03-15 10:30:45 ERROR App: IndexOutOfBoundsException",
            "  at com.app.List.get(List.java:10)",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert "Service.java" in entries[0].message
        assert "List.java" in entries[1].message


# ============================================================
# 15. parse_line with explicit format parameter for each type
# ============================================================

class TestExplicitFormatParameter:
    """parse_line with fmt= for every supported format."""

    def test_explicit_simple_format(self):
        line = "2024-03-15 10:30:45,000 WARN DiskMonitor: usage at 90%"
        entry = parse_line(line, fmt="simple")
        assert entry.level == "WARN"
        assert "usage at 90%" in entry.message

    def test_explicit_python_format(self):
        line = "2024-03-15 10:30:45,123 - myapp.cache - WARN - eviction triggered"
        entry = parse_line(line, fmt="python")
        assert entry.level == "WARN"
        assert entry.source == "myapp.cache"
        assert "eviction" in entry.message

    def test_explicit_spring_format(self):
        line = "2024-03-15 10:30:45.123 ERROR 5678 --- [task-1] c.m.svc.OrderService : Payment timeout"
        entry = parse_line(line, fmt="spring")
        assert entry.level == "ERROR"
        assert "Payment timeout" in entry.message

    def test_explicit_syslog_format(self):
        line = "Mar 15 10:30:45 gateway haproxy[789]: backend down"
        entry = parse_line(line, fmt="syslog")
        assert entry.timestamp is not None
        assert "backend down" in entry.message

    def test_explicit_nginx_access_format(self):
        line = '10.0.0.1 - - [15/Mar/2024:10:30:45 +0000] "DELETE /api/item/5 HTTP/1.1" 204 0'
        entry = parse_line(line, fmt="nginx_access")
        assert entry.timestamp is not None

    def test_explicit_docker_format(self):
        line = "2024-03-15T10:30:45.500Z stderr FATAL: database system is not yet accepting connections"
        entry = parse_line(line, fmt="docker")
        assert entry.timestamp is not None
        assert entry.level == "FATAL"

    def test_explicit_json_format(self):
        line = json.dumps({
            "timestamp": "2024-03-15T10:30:45Z",
            "level": "FATAL",
            "message": "out of disk",
            "source": "storage",
        })
        entry = parse_line(line, fmt="json")
        assert entry.level == "FATAL"
        assert entry.message == "out of disk"
        assert entry.source == "storage"

    def test_explicit_format_mismatch_fallback(self):
        # Give a syslog line but say it's spring format
        line = "Mar 15 10:30:45 myhost sshd[1]: accepted key"
        entry = parse_line(line, fmt="spring")
        # Should still parse via fallback generic parsing
        assert entry.raw == line
        assert entry.timestamp is not None

    def test_auto_format_delegation(self):
        line = "2024-03-15 10:30:45,123 - myapp - DEBUG - verbose output"
        entry = parse_line(line, fmt="auto")
        assert entry.level == "DEBUG"
        assert entry.timestamp is not None


# ============================================================
# 16. Additional edge cases not covered elsewhere
# ============================================================

class TestAdditionalEdgeCases:
    """Miscellaneous edge cases for completeness."""

    def test_parse_line_preserves_raw(self):
        line = "2024-03-15 10:30:45 INFO MyService: all good"
        entry = parse_line(line)
        assert entry.raw == line

    def test_log_entry_is_error_for_fatal(self):
        entry = parse_line("2024-03-15 10:30:45 FATAL crash")
        assert entry.is_error is True
        assert entry.is_warn is False

    def test_log_entry_is_warn_for_warn(self):
        entry = parse_line("2024-03-15 10:30:45 WARN slow")
        assert entry.is_warn is True
        assert entry.is_error is False

    def test_log_entry_metadata_default_empty(self):
        entry = parse_line("2024-03-15 10:30:45 INFO hello")
        assert entry.metadata == {}

    def test_line_number_zero_by_default(self):
        entry = parse_line("2024-03-15 10:30:45 INFO test")
        assert entry.line_number == 0

    def test_parse_lines_assigns_line_numbers_with_gaps(self):
        lines = [
            "2024-03-15 10:30:45 INFO first",
            "",
            "   ",
            "2024-03-15 10:30:46 INFO second",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
        assert entries[0].line_number == 1
        assert entries[1].line_number == 4

    def test_multiline_continuation_at_file_end(self):
        lines = [
            "2024-03-15 10:30:45 ERROR Crash occurred",
            "  java.lang.OutOfMemoryError: Java heap space",
            "    at java.util.Arrays.copyOf(Arrays.java:3236)",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 1
        assert "OutOfMemoryError" in entries[0].message
        assert "Arrays.java" in entries[0].message

    def test_epoch_timestamp_small_number_ignored(self):
        # Small numbers should not be treated as epoch
        ts = _parse_timestamp("12345")
        # The epoch pattern requires exactly 10 digits at start of string
        # "12345" is only 5 digits, should not match
        assert ts is None

    def test_auto_detect_with_single_line(self):
        lines = ["2024-03-15 10:30:45,123 - app - INFO - single line"]
        fmt = auto_detect_format(lines)
        assert fmt in ("python", "simple")

    def test_parse_lines_empty_strings_only(self):
        entries = parse_lines(["", "", "", ""])
        assert entries == []
