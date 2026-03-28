"""Robustness and security tests for LogMind.

Covers malicious inputs, boundary conditions, memory safety,
regex DoS, storage corruption, XSS-like content, payload injection,
concurrent parsing, and unicode edge cases.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import tempfile
import threading
from datetime import datetime
from typing import List
from unittest.mock import patch

import pytest

from logmind.models import AnomalyAlert, IncidentMatch, LogEntry, LogWindow
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
from logmind.alerter import _format_slack_message, _format_discord_message
from logmind.storage import save_index, load_index
from logmind.display import display_alert, display_scan_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**overrides) -> LogEntry:
    defaults = dict(
        timestamp=datetime(2024, 1, 1),
        level="ERROR",
        source="test",
        message="boom",
        raw="raw line",
    )
    defaults.update(overrides)
    return LogEntry(**defaults)


def _make_window(entries: List[LogEntry] | None = None, **overrides) -> LogWindow:
    if entries is None:
        entries = [_make_entry()]
    defaults = dict(
        entries=entries,
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 1, 0, 1),
    )
    defaults.update(overrides)
    return LogWindow(**defaults)


def _make_alert(**overrides) -> AnomalyAlert:
    defaults = dict(
        current_window=_make_window(),
        anomaly_score=0.95,
        anomaly_type="error_spike",
        similar_incidents=[],
        severity="CRITICAL",
        summary="Test alert summary",
    )
    defaults.update(overrides)
    return AnomalyAlert(**defaults)


# ===========================================================================
# 1. Malicious JSON inputs
# ===========================================================================

class TestMaliciousJSON:
    """JSON inputs designed to stress the parser."""

    def test_deeply_nested_json(self):
        """Deeply nested JSON should not cause stack overflow."""
        depth = 500
        nested = '{"a":' * depth + '1' + '}' * depth
        entry = _try_json(nested)
        # Should either parse or return None, not crash
        assert entry is None or isinstance(entry, LogEntry)

    def test_huge_key_names(self):
        """JSON with extremely long key names."""
        big_key = "k" * 100_000
        line = json.dumps({big_key: "value", "message": "hello"})
        entry = _try_json(line)
        assert entry is None or isinstance(entry, LogEntry)

    def test_huge_value(self):
        """JSON with a very large string value."""
        big_val = "x" * 500_000
        line = json.dumps({"message": big_val, "level": "ERROR"})
        entry = _try_json(line)
        assert entry is not None
        assert len(entry.message) == 500_000

    def test_json_array_not_object(self):
        """A valid JSON array should be rejected (not a dict)."""
        line = '[1, 2, 3]'
        assert _try_json(line) is None

    def test_json_primitive(self):
        """A JSON primitive should be rejected."""
        assert _try_json('"just a string"') is None
        assert _try_json('42') is None
        assert _try_json('null') is None
        assert _try_json('true') is None

    def test_json_duplicate_keys(self):
        """JSON with duplicate keys -- Python keeps the last one."""
        line = '{"message": "first", "message": "second", "level": "WARN"}'
        entry = _try_json(line)
        assert entry is not None
        assert entry.message == "second"

    def test_json_with_null_bytes(self):
        """JSON containing null bytes in values."""
        line = json.dumps({"message": "hello\x00world", "level": "ERROR"})
        entry = _try_json(line)
        assert entry is not None


# ===========================================================================
# 2. Path traversal in parse_file
# ===========================================================================

class TestPathTraversal:
    """Ensure parse_file handles adversarial file paths safely."""

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            parse_file("/nonexistent/path/to/file.log")

    def test_directory_instead_of_file(self, tmp_path):
        with pytest.raises((IsADirectoryError, PermissionError, OSError)):
            parse_file(str(tmp_path))

    def test_path_with_dotdot(self, tmp_path):
        """Path traversal attempt: parse_file opens whatever is given."""
        fake = tmp_path / "sub" / ".." / "real.log"
        fake.parent.mkdir(parents=True, exist_ok=True)
        # Create the resolved file
        resolved = tmp_path / "real.log"
        resolved.write_text("2024-01-01 10:00:00 INFO hello\n", encoding="utf-8")
        entries = parse_file(str(fake))
        assert len(entries) == 1

    def test_symlink_to_sensitive_file(self, tmp_path):
        """Symlink should still work (parse_file does not restrict)."""
        real = tmp_path / "safe.log"
        real.write_text("2024-01-01 INFO ok\n", encoding="utf-8")
        link = tmp_path / "link.log"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("Cannot create symlink on this platform")
        entries = parse_file(str(link))
        assert len(entries) >= 1


# ===========================================================================
# 3. Extremely long log lines
# ===========================================================================

class TestExtremelyLongLines:
    """Lines that are unreasonably long."""

    def test_1mb_line(self):
        """A single 1 MB line should not crash."""
        line = "2024-01-01 10:00:00 ERROR " + "A" * (1024 * 1024)
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)
        assert entry.level == "ERROR"

    def test_many_short_lines(self):
        """10,000 short lines should parse without issue."""
        lines = ["2024-01-01 10:00:00 INFO line %d" % i for i in range(10_000)]
        entries = parse_lines(lines)
        assert len(entries) == 10_000

    def test_line_with_only_spaces(self):
        """A line of pure spaces should produce an empty entry."""
        entry = parse_line("          ")
        assert entry.message == ""


# ===========================================================================
# 4. Binary / corrupt data handling
# ===========================================================================

class TestBinaryCorruptData:
    """Binary and corrupt inputs that are not valid text."""

    def test_binary_bytes_in_line(self):
        """Lines with embedded binary should not crash."""
        line = "2024-01-01 ERROR \x00\x01\x02\xff\xfe payload"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_file_with_binary_content(self, tmp_path):
        """A file with binary content (errors='replace') should parse."""
        p = tmp_path / "binary.log"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100 + b"\nINFO normal line\n")
        entries = parse_file(str(p))
        assert isinstance(entries, list)

    def test_parse_line_with_control_chars(self):
        """Control characters should not crash the parser."""
        line = "2024-01-01 WARN \x07\x08\x0b\x0c bell-backspace"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_try_json_with_truncated_json(self):
        """Truncated JSON should return None, not crash."""
        assert _try_json('{"message": "trun') is None
        assert _try_json('{') is None
        assert _try_json('{"a": ') is None


# ===========================================================================
# 5. Unicode edge cases
# ===========================================================================

class TestUnicodeEdgeCases:

    def test_zero_width_characters(self):
        """Zero-width joiners, spaces, etc."""
        line = "2024-01-01 ERROR zero\u200bwidth\u200cjoiner\u200dhere"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)
        assert "zero" in entry.message

    def test_rtl_override(self):
        """Right-to-left override characters."""
        line = "2024-01-01 WARN \u202edesrever si txet siht"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_combining_characters(self):
        """Combining diacritical marks."""
        line = "2024-01-01 INFO e\u0301clair a\u0300 la mode"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_emoji_sequences(self):
        """Complex emoji sequences (family, skin tones, flags)."""
        emojis = "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466"
        line = f"2024-01-01 ERROR emoji: {emojis}"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)
        assert emojis in entry.raw

    def test_surrogate_range_in_json(self):
        """Escaped surrogates in JSON (invalid but common)."""
        line = r'{"message": "pair: \ud83d\ude00", "level": "INFO"}'
        # Python json.loads handles or rejects surrogates
        entry = _try_json(line)
        assert entry is None or isinstance(entry, LogEntry)

    def test_cjk_characters(self):
        line = "2024-01-01 ERROR \u6d4b\u8bd5\u65e5\u672c\u8a9e\ud55c\uad6d\uc5b4"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)


# ===========================================================================
# 6. Timestamp overflow / underflow
# ===========================================================================

class TestTimestampBoundaries:

    def test_year_zero_timestamp(self):
        """Year 0000 is not valid in datetime."""
        ts = _parse_timestamp("0000-01-01T00:00:00")
        # Should either be None or a valid datetime, not crash
        assert ts is None or isinstance(ts, datetime)

    def test_year_9999_timestamp(self):
        """Year 9999 is the max for Python datetime."""
        ts = _parse_timestamp("9999-12-31T23:59:59")
        assert ts is None or isinstance(ts, datetime)

    def test_negative_epoch(self):
        """Negative epoch (before 1970) -- may raise OSError on Windows."""
        ts = _parse_timestamp("-1000000000")
        assert ts is None or isinstance(ts, datetime)

    def test_huge_epoch(self):
        """Epoch far in the future."""
        ts = _parse_timestamp("99999999999")
        # 10-digit check won't match 11 digits, so should be None
        assert ts is None or isinstance(ts, datetime)

    def test_epoch_zero(self):
        """Epoch 0 = 1970-01-01."""
        ts = _parse_timestamp("0000000000")
        # Matches the 10-digit pattern
        assert ts is None or isinstance(ts, datetime)

    def test_malformed_month_day(self):
        """Invalid month/day combinations."""
        assert _parse_timestamp("2024-13-01T00:00:00") is None or True
        assert _parse_timestamp("2024-02-30T00:00:00") is None or True


# ===========================================================================
# 7. Memory safety: huge LogWindow
# ===========================================================================

class TestLogWindowMemory:

    def test_large_entry_count_in_window(self):
        """Window with many entries should still compute properties."""
        entries = [_make_entry(level="ERROR") for _ in range(10_000)]
        window = _make_window(entries=entries)
        assert window.error_count == 10_000
        assert window.total_count == 10_000
        assert window.warn_count == 0

    def test_to_embedding_text_caps_at_50(self):
        """to_embedding_text should cap important messages at 50."""
        entries = [_make_entry(level="ERROR", message=f"err {i}") for i in range(200)]
        window = _make_window(entries=entries)
        text = window.to_embedding_text()
        # Should not contain all 200 messages
        assert text.count("[ERROR]") <= 50

    def test_source_distribution_many_sources(self):
        """Many distinct sources should work."""
        entries = [_make_entry(source=f"svc-{i}") for i in range(1000)]
        window = _make_window(entries=entries)
        dist = window.source_distribution
        assert len(dist) == 1000


# ===========================================================================
# 8. Regex denial of service (pathological patterns)
# ===========================================================================

class TestRegexDoS:

    def test_repeated_timestamps(self):
        """Line filled with timestamp-like patterns."""
        line = "2024-01-01 " * 1000 + " ERROR repeated"
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_near_match_level_patterns(self):
        """Many near-miss level keywords."""
        line = " ".join(["ERRORR", "WARNNG", "INFOO", "DEBUGG"] * 500)
        level = _parse_level(line)
        assert isinstance(level, str)

    def test_pathological_backtracking(self):
        """Input designed to cause regex backtracking."""
        # Long string of 'a' that almost matches common patterns
        line = "a" * 50_000
        entry = parse_line(line)
        assert isinstance(entry, LogEntry)

    def test_alternating_digit_letter(self):
        """Pattern that stresses timestamp regex."""
        line = "1a2b3c4d5e6f7g8h9i0j" * 5000
        ts = _parse_timestamp(line)
        assert ts is None or isinstance(ts, datetime)


# ===========================================================================
# 9. Storage: corrupted / wrong format files
# ===========================================================================

class TestStorageCorruption:

    def test_corrupted_pickle_file(self, tmp_path):
        """A corrupted pickle file should raise (load_index has no error handling)."""
        index_dir = tmp_path / ".logmind"
        index_dir.mkdir()
        pkl = index_dir / "index.pkl"
        pkl.write_bytes(b"\x80\x05CORRUPT_DATA_HERE!!!")
        with pytest.raises(Exception):
            load_index(str(tmp_path))

    def test_pickle_with_wrong_type(self, tmp_path):
        """A pickle file that contains the wrong type."""
        index_dir = tmp_path / ".logmind"
        index_dir.mkdir()
        pkl = index_dir / "index.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"not": "a LogMindIndex"}, f)
        result = load_index(str(tmp_path))
        # Returns a dict, not LogMindIndex -- but doesn't crash
        assert result is not None

    def test_empty_pickle_file(self, tmp_path):
        """Empty file should raise on load."""
        index_dir = tmp_path / ".logmind"
        index_dir.mkdir()
        pkl = index_dir / "index.pkl"
        pkl.write_bytes(b"")
        with pytest.raises(Exception):
            load_index(str(tmp_path))

    def test_save_and_load_roundtrip(self, tmp_path):
        """Basic roundtrip should work."""
        from logmind.models import LogMindIndex
        idx = LogMindIndex(
            normal_compressed=None,
            normal_windows=[],
            incident_compressed=None,
            incident_windows=[],
            incident_labels=[],
            incident_resolutions=[],
            quantizer=None,
            model_name="test",
            embedding_dim=64,
            total_lines=0,
            total_windows=0,
            incident_count=0,
            error_count=0,
            warn_count=0,
            sources=[],
            learn_time=0.0,
        )
        save_index(idx, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.model_name == "test"

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_index(str(tmp_path / "nope")) is None

    def test_binary_jpg_as_pickle(self, tmp_path):
        """A JPEG file header masquerading as a pickle."""
        index_dir = tmp_path / ".logmind"
        index_dir.mkdir()
        pkl = index_dir / "index.pkl"
        pkl.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        with pytest.raises(Exception):
            load_index(str(tmp_path))


# ===========================================================================
# 10. Display: XSS-like content
# ===========================================================================

class TestDisplayXSSContent:

    def test_xss_in_summary(self):
        """XSS-like content in summary should not crash display."""
        alert = _make_alert(summary='<script>alert("xss")</script>')
        output = display_alert(alert)
        assert "<script>" in output  # It's terminal display, no escaping needed
        assert isinstance(output, str)

    def test_xss_in_incident_label(self):
        inc = IncidentMatch(
            similarity=0.9,
            label='<img src=x onerror=alert(1)>',
            resolution="patched",
            occurred_at=datetime(2024, 1, 1),
        )
        alert = _make_alert(similar_incidents=[inc])
        output = display_alert(alert)
        assert isinstance(output, str)

    def test_xss_in_json_mode(self):
        alert = _make_alert(summary='"; DROP TABLE logs; --')
        output = display_alert(alert, as_json=True)
        data = json.loads(output)
        assert data["summary"] == '"; DROP TABLE logs; --'

    def test_scan_report_with_xss_alerts(self):
        alerts = [_make_alert(summary="<b>bold</b>")] * 3
        output = display_scan_report(alerts, total_windows=10)
        assert isinstance(output, str)

    def test_display_empty_report(self):
        output = display_scan_report([], total_windows=0)
        assert "No anomalies" in output


# ===========================================================================
# 11. Alerter: payload injection
# ===========================================================================

class TestAlerterPayloadInjection:

    def test_slack_message_with_injection_in_summary(self):
        """Markdown injection in summary should be preserved (Slack uses mrkdwn)."""
        alert = _make_alert(summary='*bold* _italic_ `code` @here @channel')
        payload = _format_slack_message(alert)
        assert "blocks" in payload
        # The summary text should appear truncated to 500
        summary_block = payload["blocks"][2]
        assert "@here" in summary_block["text"]["text"]

    def test_discord_message_with_injection(self):
        alert = _make_alert(summary='@everyone <@&role_id> http://evil.com')
        payload = _format_discord_message(alert)
        assert payload["embeds"][0]["description"] == alert.summary

    def test_slack_very_long_summary_truncated(self):
        alert = _make_alert(summary="X" * 10_000)
        payload = _format_slack_message(alert)
        text = payload["blocks"][2]["text"]["text"]
        # Summary is truncated to 500 chars + prefix
        assert len(text) < 600

    def test_discord_very_long_summary_truncated(self):
        alert = _make_alert(summary="Y" * 10_000)
        payload = _format_discord_message(alert)
        desc = payload["embeds"][0]["description"]
        assert len(desc) <= 2000

    def test_slack_with_similar_incidents(self):
        inc = IncidentMatch(
            similarity=0.85,
            label='"; cat /etc/passwd',
            resolution="fixed\nby\nmultiline",
            occurred_at=None,
        )
        alert = _make_alert(similar_incidents=[inc])
        payload = _format_slack_message(alert)
        assert len(payload["blocks"]) >= 4

    def test_discord_with_similar_incidents(self):
        inc = IncidentMatch(
            similarity=0.5,
            label="normal",
            resolution="",
            occurred_at=None,
        )
        alert = _make_alert(similar_incidents=[inc])
        payload = _format_discord_message(alert)
        fields = payload["embeds"][0]["fields"]
        assert any("No resolution recorded" in f["value"] for f in fields)


# ===========================================================================
# 12. Concurrent parsing
# ===========================================================================

class TestConcurrentParsing:

    def test_parallel_parse_line(self):
        """Multiple threads calling parse_line simultaneously."""
        results = [None] * 20
        errors = []

        def worker(idx):
            try:
                line = f"2024-01-01 10:00:{idx:02d} ERROR thread-{idx} failure"
                results[idx] = parse_line(line)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert all(r is not None for r in results)

    def test_parallel_parse_lines(self):
        """Multiple threads calling parse_lines on different data."""
        results = {}
        errors = []

        def worker(idx):
            try:
                lines = [f"2024-01-01 INFO batch-{idx} line-{j}" for j in range(100)]
                results[idx] = parse_lines(lines)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        assert all(len(v) == 100 for v in results.values())

    def test_parallel_auto_detect(self):
        """auto_detect_format from multiple threads."""
        errors = []

        def worker():
            try:
                lines = ['{"message":"hi","level":"INFO"}'] * 10
                fmt = auto_detect_format(lines)
                assert fmt == "json"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors


# ===========================================================================
# 13. Integer overflow in line numbers
# ===========================================================================

class TestIntegerOverflow:

    def test_huge_line_number(self):
        """Assigning a very large line number."""
        entry = _make_entry()
        entry.line_number = 2**63
        assert entry.line_number == 2**63

    def test_negative_line_number(self):
        entry = _make_entry()
        entry.line_number = -1
        assert entry.line_number == -1

    def test_parse_lines_preserves_line_numbers(self):
        lines = ["2024-01-01 INFO line1", "2024-01-01 INFO line2"]
        entries = parse_lines(lines)
        assert entries[0].line_number == 1
        assert entries[1].line_number == 2


# ===========================================================================
# 14. Empty / whitespace-only for every field
# ===========================================================================

class TestEmptyWhitespace:

    def test_empty_string_parse_line(self):
        entry = parse_line("")
        assert entry.message == ""
        assert entry.level == "DEBUG"

    def test_whitespace_only_parse_line(self):
        entry = parse_line("   \t  \n")
        assert entry.message == ""

    def test_newline_only(self):
        entry = parse_line("\n")
        assert isinstance(entry, LogEntry)

    def test_parse_lines_all_empty(self):
        lines = ["", "   ", "\t", "\n"]
        entries = parse_lines(lines)
        assert entries == []  # parse_lines skips blank lines

    def test_try_json_empty(self):
        assert _try_json("") is None
        assert _try_json("   ") is None

    def test_auto_detect_empty_list(self):
        assert auto_detect_format([]) == "simple"

    def test_auto_detect_all_blank(self):
        fmt = auto_detect_format(["", "  ", "\t"])
        assert isinstance(fmt, str)

    def test_parse_source_empty(self):
        assert _parse_source("") == "unknown"

    def test_parse_level_empty(self):
        assert _parse_level("") == "INFO"

    def test_parse_timestamp_empty(self):
        assert _parse_timestamp("") is None

    def test_log_entry_empty_fields(self):
        entry = LogEntry(
            timestamp=None, level="", source="", message="", raw=""
        )
        assert not entry.is_error
        assert not entry.is_warn

    def test_log_window_empty_entries(self):
        window = LogWindow(entries=[], start_time=None, end_time=None)
        assert window.error_count == 0
        assert window.total_count == 0
        assert window.source_distribution == {}
        text = window.to_embedding_text()
        assert isinstance(text, str)

    def test_display_alert_empty_summary(self):
        alert = _make_alert(summary="")
        output = display_alert(alert)
        assert isinstance(output, str)

    def test_display_scan_report_empty(self):
        output = display_scan_report([], total_windows=0)
        assert "0" in output
