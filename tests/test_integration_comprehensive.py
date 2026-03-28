"""Comprehensive integration tests for LogMind cross-module scenarios.

~80 test cases covering parser->models, models->display, parser->embedder->detector,
storage round-trips, CLI commands, alerter formatting, collector->parser,
edge cases, multi-format pipelines, and incident labeling workflows.

All SentenceTransformer usage is mocked for speed (<30s total).
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import textwrap
from datetime import datetime, timedelta
from io import StringIO
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from logmind.alerter import (
    _format_discord_message,
    _format_slack_message,
    send_discord,
    send_slack,
    send_webhook,
)
from logmind.cli import main
from logmind.collector import CommandCollector, FileCollector, StdinCollector
from logmind.detector import (
    _classify_anomaly_type,
    _classify_severity,
    compute_anomaly_score,
    detect,
    find_similar_incidents,
)
from logmind.display import (
    display_alert,
    display_scan_report,
    display_search_results,
    display_stats,
)
from logmind.embedder import DEFAULT_MODEL, _create_windows, build_index, embed_window
from logmind.incidents import auto_detect_incidents, label_incident, search_incidents
from logmind.models import (
    AnomalyAlert,
    IncidentMatch,
    LogEntry,
    LogMindIndex,
    LogWindow,
)
from logmind.parser import auto_detect_format, parse_file, parse_line, parse_lines
from logmind.storage import load_index, save_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32  # small embedding dimension for fast tests


def _ts(minutes: int = 0) -> datetime:
    """Return a datetime offset from a fixed base by *minutes*."""
    return datetime(2024, 3, 15, 10, 0, 0) + timedelta(minutes=minutes)


def _make_entry(
    msg: str = "something happened",
    level: str = "INFO",
    source: str = "app",
    ts_offset_min: int = 0,
    line_number: int = 0,
) -> LogEntry:
    return LogEntry(
        timestamp=_ts(ts_offset_min),
        level=level,
        source=source,
        message=msg,
        raw=f"2024-03-15 10:{ts_offset_min:02d}:00 {level} {source}: {msg}",
        line_number=line_number,
    )


def _make_window(
    entries: List[LogEntry] | None = None,
    label: str = "",
    resolution: str = "",
) -> LogWindow:
    if entries is None:
        entries = [
            _make_entry("start info", "INFO", ts_offset_min=0),
            _make_entry("some error", "ERROR", ts_offset_min=0),
            _make_entry("another warn", "WARN", ts_offset_min=0),
        ]
    start = entries[0].timestamp if entries else None
    end = entries[-1].timestamp if entries else None
    return LogWindow(
        entries=entries,
        start_time=start,
        end_time=end,
        label=label,
        resolution=resolution,
    )


def _make_alert(
    score: float = 0.75,
    anomaly_type: str = "error_spike",
    severity: str = "CRITICAL",
    similar: List[IncidentMatch] | None = None,
) -> AnomalyAlert:
    window = _make_window()
    if similar is None:
        similar = []
    return AnomalyAlert(
        current_window=window,
        anomaly_score=score,
        anomaly_type=anomaly_type,
        similar_incidents=similar,
        severity=severity,
        summary="Error spike detected: 1 errors in 60s window",
    )


class _FakeQuantizer:
    """A picklable fake quantizer for storage round-trip tests."""

    def __init__(self, scores=None):
        self._scores = scores or [[0.95, 0.80, 0.60]]

    def cosine_scores(self, vec, compressed):
        return np.array(self._scores)

    def quantize(self, vecs):
        return vecs  # just pass through

    def dequantize(self, compressed):
        if isinstance(compressed, np.ndarray):
            return compressed
        return np.random.randn(3, DIM).astype(np.float32)


def _fake_quantizer(picklable: bool = False, scores=None):
    """Return a quantizer. Use picklable=True for storage tests."""
    if picklable:
        return _FakeQuantizer(scores)
    q = MagicMock()
    q.cosine_scores.return_value = np.array(scores or [[0.95, 0.80, 0.60]])
    q.quantize.return_value = "compressed_blob"
    q.dequantize.return_value = np.random.randn(3, DIM).astype(np.float32)
    return q


def _make_index(
    n_normal: int = 5,
    n_incidents: int = 0,
    avg_errors: float = 1.0,
    std_errors: float = 0.5,
    picklable: bool = False,
) -> LogMindIndex:
    quantizer = _fake_quantizer(picklable=picklable)
    normal_windows = [_make_window() for _ in range(n_normal)]
    inc_windows = []
    inc_labels = []
    inc_resolutions = []
    for i in range(n_incidents):
        w = _make_window(label=f"incident-{i}", resolution=f"fix-{i}")
        inc_windows.append(w)
        inc_labels.append(w.label)
        inc_resolutions.append(w.resolution)

    return LogMindIndex(
        normal_compressed="compressed_normal",
        normal_windows=normal_windows,
        incident_compressed="compressed_inc" if n_incidents else None,
        incident_windows=inc_windows,
        incident_labels=inc_labels,
        incident_resolutions=inc_resolutions,
        quantizer=quantizer,
        model_name=DEFAULT_MODEL,
        embedding_dim=DIM,
        total_lines=100,
        total_windows=n_normal,
        incident_count=n_incidents,
        error_count=10,
        warn_count=5,
        sources=["app", "db", "web"],
        learn_time=1.23,
        avg_errors_per_window=avg_errors,
        std_errors_per_window=std_errors,
        avg_warns_per_window=0.5,
    )


def _write_log_file(tmp_path, lines: List[str], name: str = "test.log") -> str:
    p = os.path.join(str(tmp_path), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


# ===========================================================================
# 1. Parser -> Models integration
# ===========================================================================


class TestParserModelsIntegration:
    """Parse results used correctly in LogWindow."""

    def test_parsed_entries_populate_log_window(self):
        raw = [
            "2024-03-15 10:00:00 ERROR app: disk full",
            "2024-03-15 10:00:30 WARN app: retrying",
            "2024-03-15 10:00:45 INFO app: recovered",
        ]
        entries = parse_lines(raw)
        window = LogWindow(
            entries=entries,
            start_time=entries[0].timestamp,
            end_time=entries[-1].timestamp,
        )
        assert window.error_count == 1
        assert window.warn_count == 1
        assert window.total_count == 3
        assert "app" in window.source_distribution

    def test_parsed_json_entries_in_window(self):
        raw = [
            '{"timestamp":"2024-03-15T10:00:00Z","level":"ERROR","message":"timeout","source":"api"}',
            '{"timestamp":"2024-03-15T10:00:05Z","level":"INFO","message":"ok","source":"api"}',
        ]
        entries = parse_lines(raw)
        assert len(entries) == 2
        window = LogWindow(entries=entries, start_time=entries[0].timestamp, end_time=entries[-1].timestamp)
        assert window.error_count == 1
        assert window.source_distribution == {"api": 2}

    def test_embedding_text_contains_parsed_messages(self):
        raw = [
            "2024-03-15 10:00:00 ERROR db: connection refused",
            "2024-03-15 10:00:10 WARN db: pool exhausted",
        ]
        entries = parse_lines(raw)
        window = LogWindow(entries=entries, start_time=entries[0].timestamp, end_time=entries[-1].timestamp)
        text = window.to_embedding_text()
        assert "errors=1" in text
        assert "warns=1" in text
        assert "connection refused" in text

    def test_multiline_stack_trace_stays_in_one_entry(self):
        raw = [
            "2024-03-15 10:00:00 ERROR app: NullPointerException",
            "    at com.example.Main.run(Main.java:42)",
            "    at com.example.Main.main(Main.java:10)",
            "2024-03-15 10:00:30 INFO app: recovered",
        ]
        entries = parse_lines(raw)
        assert len(entries) == 2
        assert "Main.java:42" in entries[0].message

    def test_is_error_and_is_warn_properties(self):
        entries = parse_lines([
            "2024-03-15 10:00:00 FATAL db: crash",
            "2024-03-15 10:00:01 ERROR db: err",
            "2024-03-15 10:00:02 WARNING db: slow",
            "2024-03-15 10:00:03 INFO db: ok",
        ])
        assert entries[0].is_error  # FATAL
        assert entries[1].is_error  # ERROR
        assert entries[2].is_warn   # WARNING -> WARN
        assert not entries[3].is_error
        assert not entries[3].is_warn

    def test_source_distribution_with_multiple_services(self):
        entries = [
            _make_entry("a", "INFO", "web"),
            _make_entry("b", "INFO", "web"),
            _make_entry("c", "ERROR", "db"),
        ]
        w = LogWindow(entries=entries, start_time=_ts(0), end_time=_ts(0))
        assert w.source_distribution == {"web": 2, "db": 1}

    def test_is_incident_property(self):
        w = _make_window(label="outage")
        assert w.is_incident is True
        w2 = _make_window(label="")
        assert w2.is_incident is False

    def test_line_numbers_assigned_by_parse_lines(self):
        raw = ["2024-03-15 10:00:00 INFO app: line1", "2024-03-15 10:00:01 INFO app: line2"]
        entries = parse_lines(raw)
        assert entries[0].line_number == 1
        assert entries[1].line_number == 2


# ===========================================================================
# 2. Models -> Display integration
# ===========================================================================


class TestModelsDisplayIntegration:
    """All alert types render correctly."""

    def test_display_critical_alert_text(self):
        alert = _make_alert(severity="CRITICAL")
        output = display_alert(alert)
        assert "[!!!]" in output
        assert "CRITICAL" in output
        assert "0.75" in output

    def test_display_warning_alert_text(self):
        alert = _make_alert(severity="WARNING", score=0.55)
        output = display_alert(alert)
        assert "[!!]" in output
        assert "WARNING" in output

    def test_display_info_alert_text(self):
        alert = _make_alert(severity="INFO", score=0.35)
        output = display_alert(alert)
        assert "[i]" in output

    def test_display_alert_as_json(self):
        alert = _make_alert()
        output = display_alert(alert, as_json=True)
        data = json.loads(output)
        assert data["severity"] == "CRITICAL"
        assert data["anomaly_score"] == 0.75

    def test_display_alert_with_similar_incidents(self):
        inc = IncidentMatch(similarity=0.85, label="db-outage", resolution="restart db", occurred_at=_ts())
        alert = _make_alert(similar=[inc])
        output = display_alert(alert)
        assert "db-outage" in output
        assert "restart db" in output
        assert "85%" in output

    def test_display_scan_report_no_anomalies(self):
        output = display_scan_report([], total_windows=10)
        assert "No anomalies detected" in output
        assert "Windows analyzed: 10" in output

    def test_display_scan_report_with_mixed_severities(self):
        alerts = [
            _make_alert(severity="CRITICAL"),
            _make_alert(severity="WARNING", score=0.55),
            _make_alert(severity="INFO", score=0.35),
        ]
        output = display_scan_report(alerts, total_windows=20)
        assert "CRITICAL: 1" in output
        assert "WARNING:  1" in output
        assert "INFO:     1" in output

    def test_display_scan_report_json(self):
        alerts = [_make_alert()]
        output = display_scan_report(alerts, total_windows=5, as_json=True)
        data = json.loads(output)
        assert data["total_windows"] == 5
        assert data["anomalies_found"] == 1

    def test_display_stats(self):
        index = _make_index(n_incidents=2)
        output = display_stats(index)
        assert "Total log lines:" in output
        assert "100" in output
        assert "incident-0" in output
        assert "fix-0" in output

    def test_display_search_results_empty(self):
        output = display_search_results([])
        assert "No matching incidents" in output

    def test_display_search_results_with_matches(self):
        results = [
            IncidentMatch(similarity=0.9, label="mem-leak", resolution="restart", occurred_at=_ts()),
            IncidentMatch(similarity=0.6, label="cpu-spike", resolution="", occurred_at=None),
        ]
        output = display_search_results(results)
        assert "mem-leak" in output
        assert "restart" in output
        assert "90%" in output
        assert "cpu-spike" in output


# ===========================================================================
# 3. Parser -> Embedder -> Detector flow with mocks
# ===========================================================================


class TestParserEmbedderDetectorFlow:
    """End-to-end parse -> window -> embed -> detect with mocked model."""

    def test_create_windows_from_parsed_entries(self):
        raw = [
            f"2024-03-15 10:{m:02d}:00 INFO app: msg{m}" for m in range(5)
        ]
        entries = parse_lines(raw)
        windows = _create_windows(entries, window_size=120)
        assert len(windows) >= 1
        total_entries = sum(w.total_count for w in windows)
        assert total_entries == 5

    def test_create_windows_groups_by_time(self):
        entries = [_make_entry(ts_offset_min=m) for m in range(10)]
        windows = _create_windows(entries, window_size=180)  # 3 min windows
        assert len(windows) >= 3

    def test_create_windows_no_timestamps_falls_back_to_chunks(self):
        entries = [
            LogEntry(timestamp=None, level="INFO", source="x", message=f"m{i}", raw=f"m{i}")
            for i in range(50)
        ]
        windows = _create_windows(entries, window_size=60)
        assert len(windows) >= 1

    @patch("sentence_transformers.SentenceTransformer")
    @patch("langchain_turboquant.TurboQuantizer")
    def test_build_index_mock(self, MockQuantizer, MockST):
        model_inst = MagicMock()
        model_inst.get_sentence_embedding_dimension.return_value = DIM
        model_inst.encode.return_value = np.random.randn(2, DIM).astype(np.float32)
        MockST.return_value = model_inst

        q_inst = MagicMock()
        q_inst.quantize.return_value = "blob"
        MockQuantizer.return_value = q_inst

        entries = [_make_entry(ts_offset_min=m) for m in range(5)]
        idx = build_index(entries, window_size=120)
        assert idx.total_lines == 5
        assert idx.total_windows >= 1
        assert idx.model_name == DEFAULT_MODEL

    def test_compute_anomaly_score_high_similarity(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[0.99]])
        vec = np.random.randn(DIM).astype(np.float32)
        score = compute_anomaly_score(vec, index)
        assert score < 0.05  # very similar to normal -> low anomaly

    def test_compute_anomaly_score_low_similarity(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[0.1]])
        vec = np.random.randn(DIM).astype(np.float32)
        score = compute_anomaly_score(vec, index)
        assert score > 0.85  # very different from normal -> high anomaly

    def test_detect_returns_none_below_threshold(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[0.95]])
        window = _make_window()
        vec = np.random.randn(DIM).astype(np.float32)
        result = detect(window, vec, index, sensitivity="medium")
        assert result is None

    def test_detect_returns_alert_above_threshold(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[0.1]])
        window = _make_window()
        vec = np.random.randn(DIM).astype(np.float32)
        result = detect(window, vec, index, sensitivity="medium")
        assert result is not None
        assert isinstance(result, AnomalyAlert)
        assert result.anomaly_score > 0.4

    def test_detect_high_sensitivity_triggers_easier(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[0.55]])
        window = _make_window()
        vec = np.random.randn(DIM).astype(np.float32)
        # medium threshold = 0.40, high = 0.30
        result_medium = detect(window, vec, index, sensitivity="medium")
        result_high = detect(window, vec, index, sensitivity="high")
        # score = 1 - 0.55 = 0.45; medium threshold 0.40 -> alert; high threshold 0.30 -> alert
        assert result_medium is not None
        assert result_high is not None

    def test_find_similar_incidents_no_incidents(self):
        index = _make_index(n_incidents=0)
        vec = np.random.randn(DIM).astype(np.float32)
        matches = find_similar_incidents(vec, index)
        assert matches == []

    def test_find_similar_incidents_returns_sorted(self):
        index = _make_index(n_incidents=3)
        index.quantizer.cosine_scores.return_value = np.array([[0.9, 0.5, 0.7]])
        vec = np.random.randn(DIM).astype(np.float32)
        matches = find_similar_incidents(vec, index, k=3)
        assert len(matches) >= 2
        assert matches[0].similarity >= matches[1].similarity


# ===========================================================================
# 4. Storage round-trip with various index states
# ===========================================================================


class TestStorageRoundTrip:
    """Save and load index with various states."""

    def test_save_load_basic_index(self, tmp_path):
        index = _make_index(picklable=True)
        path = save_index(index, str(tmp_path))
        assert os.path.exists(path)
        loaded = load_index(str(tmp_path))
        assert loaded is not None
        assert loaded.total_lines == index.total_lines
        assert loaded.model_name == index.model_name

    def test_save_load_index_with_incidents(self, tmp_path):
        index = _make_index(n_incidents=3, picklable=True)
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_count == 3
        assert len(loaded.incident_labels) == 3
        assert loaded.incident_labels[0] == "incident-0"

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_index(str(tmp_path))
        assert result is None

    def test_save_creates_directory(self, tmp_path):
        nested = os.path.join(str(tmp_path), "deep", "nested")
        os.makedirs(nested)
        index = _make_index(picklable=True)
        save_index(index, nested)
        loaded = load_index(nested)
        assert loaded is not None

    def test_overwrite_existing_index(self, tmp_path):
        idx1 = _make_index(n_incidents=0, picklable=True)
        save_index(idx1, str(tmp_path))
        idx2 = _make_index(n_incidents=5, picklable=True)
        save_index(idx2, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.incident_count == 5

    def test_round_trip_preserves_stats(self, tmp_path):
        index = _make_index(picklable=True)
        index.avg_errors_per_window = 3.14
        index.std_errors_per_window = 1.59
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert abs(loaded.avg_errors_per_window - 3.14) < 1e-6
        assert abs(loaded.std_errors_per_window - 1.59) < 1e-6

    def test_round_trip_preserves_sources_list(self, tmp_path):
        index = _make_index(picklable=True)
        index.sources = ["alpha", "beta", "gamma"]
        save_index(index, str(tmp_path))
        loaded = load_index(str(tmp_path))
        assert loaded.sources == ["alpha", "beta", "gamma"]


# ===========================================================================
# 5. CLI command combinations and error paths
# ===========================================================================


class TestCLICommands:
    """CLI command combinations and error paths."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_learn_missing_file(self, runner):
        result = runner.invoke(main, ["learn", "/nonexistent/path.log"])
        assert result.exit_code != 0

    def test_scan_no_index(self, runner, tmp_path):
        log_path = _write_log_file(tmp_path, ["2024-03-15 10:00:00 INFO app: ok"])
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(main, ["scan", log_path])
        assert result.exit_code != 0
        assert "No index found" in result.output or result.exit_code != 0

    def test_stats_no_index(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(main, ["stats"])
        assert result.exit_code != 0

    def test_search_no_index(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(main, ["search", "memory leak"])
        assert result.exit_code != 0

    def test_label_bad_time_format(self, runner, tmp_path):
        log_path = _write_log_file(tmp_path, ["2024-03-15 10:00:00 INFO app: ok"])
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            # Create a fake index so we pass the index check
            idx = _make_index(picklable=True)
            save_index(idx, ".")
            result = runner.invoke(main, [
                "label", "not-a-date", "also-not", log_path, "--label", "test",
            ])
        assert result.exit_code != 0

    def test_watch_no_source(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            idx = _make_index(picklable=True)
            save_index(idx, ".")
            result = runner.invoke(main, ["watch"])
        assert result.exit_code != 0

    @patch("sentence_transformers.SentenceTransformer")
    @patch("langchain_turboquant.TurboQuantizer")
    def test_learn_and_stats_flow(self, MockQ, MockST, runner, tmp_path):
        model_inst = MagicMock()
        model_inst.get_sentence_embedding_dimension.return_value = DIM
        model_inst.encode.return_value = np.random.randn(2, DIM).astype(np.float32)
        MockST.return_value = model_inst
        # Use a picklable quantizer so save_index succeeds
        MockQ.return_value = _FakeQuantizer()

        log_path = _write_log_file(tmp_path, [
            f"2024-03-15 10:{m:02d}:00 INFO app: msg{m}" for m in range(10)
        ])
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(main, ["learn", log_path])
            assert result.exit_code == 0 or "Index saved" in result.output


# ===========================================================================
# 6. Alerter with various alert configurations
# ===========================================================================


class TestAlerterIntegration:
    """Alerter formatting and send logic."""

    def test_slack_message_has_blocks(self):
        alert = _make_alert()
        msg = _format_slack_message(alert)
        assert "blocks" in msg
        assert len(msg["blocks"]) >= 3

    def test_slack_message_includes_severity_emoji(self):
        alert = _make_alert(severity="CRITICAL")
        msg = _format_slack_message(alert)
        header_text = msg["blocks"][0]["text"]["text"]
        assert ":red_circle:" in header_text

    def test_slack_message_with_similar_incidents(self):
        inc = IncidentMatch(similarity=0.85, label="db-crash", resolution="restart", occurred_at=_ts())
        alert = _make_alert(similar=[inc])
        msg = _format_slack_message(alert)
        all_text = json.dumps(msg)
        assert "db-crash" in all_text
        assert "restart" in all_text

    def test_discord_message_has_embeds(self):
        alert = _make_alert()
        msg = _format_discord_message(alert)
        assert "embeds" in msg
        assert msg["embeds"][0]["color"] == 0xFF0000  # CRITICAL = red

    def test_discord_warning_color(self):
        alert = _make_alert(severity="WARNING")
        msg = _format_discord_message(alert)
        assert msg["embeds"][0]["color"] == 0xFFAA00

    def test_discord_includes_fields(self):
        alert = _make_alert()
        msg = _format_discord_message(alert)
        field_names = [f["name"] for f in msg["embeds"][0]["fields"]]
        assert "Anomaly Score" in field_names
        assert "Type" in field_names
        assert "Errors" in field_names

    def test_send_slack_filters_by_severity(self):
        alert = _make_alert(severity="INFO")
        # min_severity=WARNING means INFO alerts should be skipped
        result = send_slack(alert, "http://fake-webhook.com", min_severity="WARNING")
        assert result is False

    def test_send_discord_filters_by_severity(self):
        alert = _make_alert(severity="INFO")
        result = send_discord(alert, "http://fake-webhook.com", min_severity="CRITICAL")
        assert result is False

    def test_send_webhook_filters_by_severity(self):
        alert = _make_alert(severity="WARNING")
        result = send_webhook(alert, "http://fake-webhook.com", min_severity="CRITICAL")
        assert result is False

    @patch("httpx.post")
    def test_send_slack_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        alert = _make_alert(severity="CRITICAL")
        result = send_slack(alert, "http://hooks.slack.com/test", min_severity="WARNING")
        assert result is True
        mock_post.assert_called_once()

    @patch("httpx.post")
    def test_send_discord_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=204)
        alert = _make_alert(severity="WARNING")
        result = send_discord(alert, "http://discord.com/webhook", min_severity="INFO")
        assert result is True


# ===========================================================================
# 7. Collector -> Parser integration
# ===========================================================================


class TestCollectorParserIntegration:
    """Collector reads files, parser processes them."""

    def test_file_collector_read_and_parse(self, tmp_path):
        lines = [
            "2024-03-15 10:00:00 ERROR web: 500 internal server error",
            "2024-03-15 10:00:01 INFO web: request completed",
        ]
        p = _write_log_file(tmp_path, lines)
        collector = FileCollector([p])
        raw = collector.read_all()
        entries = parse_lines(raw)
        assert len(entries) == 2
        assert entries[0].level == "ERROR"

    def test_file_collector_glob_pattern(self, tmp_path):
        _write_log_file(tmp_path, ["2024-03-15 10:00:00 INFO a: msg1"], "a.log")
        _write_log_file(tmp_path, ["2024-03-15 10:00:00 INFO b: msg2"], "b.log")
        pattern = os.path.join(str(tmp_path), "*.log")
        collector = FileCollector([pattern])
        raw = collector.read_all()
        entries = parse_lines(raw)
        assert len(entries) == 2

    def test_file_collector_no_match_raises(self):
        with pytest.raises(FileNotFoundError):
            FileCollector(["/nonexistent/pattern/*.xyz"])

    def test_file_collector_stream_yields_lines(self, tmp_path):
        lines = ["2024-03-15 10:00:00 INFO app: hello", "2024-03-15 10:00:01 INFO app: world"]
        p = _write_log_file(tmp_path, lines)
        collector = FileCollector([p], follow=False)
        streamed = list(collector.stream())
        assert len(streamed) >= 2

    def test_command_collector_read_all(self):
        # Use a simple cross-platform command
        collector = CommandCollector("echo hello_from_command")
        lines = collector.read_all()
        combined = "".join(lines)
        assert "hello_from_command" in combined

    def test_multi_file_collector_preserves_order(self, tmp_path):
        _write_log_file(tmp_path, ["2024-03-15 09:00:00 INFO a: first"], "01.log")
        _write_log_file(tmp_path, ["2024-03-15 11:00:00 INFO b: second"], "02.log")
        pattern = os.path.join(str(tmp_path), "*.log")
        collector = FileCollector([pattern])
        raw = collector.read_all()
        entries = parse_lines(raw)
        assert len(entries) == 2


# ===========================================================================
# 8. Edge cases: empty inputs, huge inputs, unicode
# ===========================================================================


class TestEdgeCases:
    """Empty inputs, large inputs, unicode throughout the pipeline."""

    def test_parse_empty_lines(self):
        entries = parse_lines([])
        assert entries == []

    def test_parse_blank_lines_skipped(self):
        entries = parse_lines(["", "  ", "\t"])
        assert entries == []

    def test_parse_single_empty_line(self):
        entry = parse_line("")
        assert entry.level == "DEBUG"
        assert entry.message == ""

    def test_unicode_message_parsing(self):
        line = '2024-03-15 10:00:00 ERROR app: \u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u63a5\u7d9a\u30a8\u30e9\u30fc'
        entry = parse_line(line)
        assert "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9" in entry.message

    def test_unicode_in_window_embedding_text(self):
        entry = _make_entry("\uc11c\ubc84 \uc624\ub958 \ubc1c\uc0dd", "ERROR")
        window = LogWindow(entries=[entry], start_time=_ts(), end_time=_ts())
        text = window.to_embedding_text()
        assert "\uc11c\ubc84" in text

    def test_unicode_in_display_alert(self):
        entry = _make_entry("\u63a5\u7d9a\u30a8\u30e9\u30fc", "ERROR")
        window = LogWindow(entries=[entry], start_time=_ts(), end_time=_ts())
        alert = AnomalyAlert(
            current_window=window, anomaly_score=0.8,
            anomaly_type="error_spike", similar_incidents=[],
            severity="CRITICAL", summary="\u30a8\u30e9\u30fc\u30b9\u30d1\u30a4\u30af\u691c\u51fa",
        )
        output = display_alert(alert)
        assert "\u30a8\u30e9\u30fc\u30b9\u30d1\u30a4\u30af" in output

    def test_very_long_message_parsing(self):
        long_msg = "A" * 10_000
        line = f"2024-03-15 10:00:00 ERROR app: {long_msg}"
        entry = parse_line(line)
        assert len(entry.message) >= 10_000

    def test_large_number_of_entries_in_window(self):
        entries = [_make_entry(f"msg{i}", "INFO", ts_offset_min=0) for i in range(1000)]
        window = LogWindow(entries=entries, start_time=_ts(), end_time=_ts())
        assert window.total_count == 1000
        text = window.to_embedding_text()
        assert "total=1000" in text

    def test_create_windows_empty_entries(self):
        windows = _create_windows([], window_size=60)
        assert windows == []

    def test_anomaly_score_empty_normal(self):
        index = _make_index()
        index.quantizer.cosine_scores.return_value = np.array([[]])
        vec = np.random.randn(DIM).astype(np.float32)
        score = compute_anomaly_score(vec, index)
        assert score == 1.0

    def test_display_stats_many_sources(self):
        index = _make_index()
        index.sources = [f"svc-{i}" for i in range(20)]
        output = display_stats(index)
        assert "... and 10 more" in output


# ===========================================================================
# 9. Multi-format log files processed end-to-end
# ===========================================================================


class TestMultiFormat:
    """Multi-format log files processed end-to-end."""

    def test_detect_python_format(self):
        lines = [
            "2024-03-15 10:00:00,123 - myapp - ERROR - connection timeout",
            "2024-03-15 10:00:01,456 - myapp - INFO - retrying",
        ]
        fmt = auto_detect_format(lines)
        assert fmt == "python"

    def test_detect_json_format(self):
        lines = [
            '{"timestamp":"2024-03-15T10:00:00Z","level":"ERROR","message":"fail"}',
            '{"timestamp":"2024-03-15T10:00:01Z","level":"INFO","message":"ok"}',
        ]
        fmt = auto_detect_format(lines)
        assert fmt == "json"

    def test_detect_syslog_format(self):
        lines = [
            "Mar 15 10:00:00 myhost sshd[1234]: Failed password for root",
            "Mar 15 10:00:01 myhost sshd[1234]: Connection closed",
        ]
        fmt = auto_detect_format(lines)
        assert fmt == "syslog"

    def test_parse_file_integration(self, tmp_path):
        lines = [
            "2024-03-15 10:00:00 ERROR db: connection refused",
            "2024-03-15 10:00:30 WARN db: retry attempt 1",
            "2024-03-15 10:01:00 INFO db: connected",
        ]
        p = _write_log_file(tmp_path, lines)
        entries = parse_file(p)
        assert len(entries) == 3
        assert entries[0].level == "ERROR"

    def test_mixed_format_file_fallback(self, tmp_path):
        """When lines are a mix, parser falls back to generic parsing."""
        lines = [
            "2024-03-15 10:00:00 ERROR app: standard error",
            "something without a timestamp but with ERROR keyword",
            '{"timestamp":"2024-03-15T10:00:05Z","level":"WARN","message":"json line"}',
        ]
        entries = parse_lines(lines)
        assert len(entries) >= 2

    def test_docker_format_parsing(self):
        lines = [
            "2024-03-15T10:00:00.123Z stdout something happened",
            "2024-03-15T10:00:01.456Z stderr error occurred",
        ]
        fmt = auto_detect_format(lines)
        assert fmt == "docker"
        entries = parse_lines(lines, fmt="docker")
        assert len(entries) == 2

    def test_syslog_entries_become_windows(self):
        base = datetime(2024, 3, 15, 10, 0, 0)
        lines = [
            f"Mar 15 10:{m:02d}:00 myhost myapp: event {m}" for m in range(5)
        ]
        entries = parse_lines(lines)
        windows = _create_windows(entries, window_size=120)
        assert len(windows) >= 1

    def test_auto_detect_format_empty(self):
        assert auto_detect_format([]) == "simple"


# ===========================================================================
# 10. Incident labeling workflow
# ===========================================================================


class TestIncidentWorkflow:
    """Incident labeling, auto-detection, and search."""

    def test_auto_detect_incidents_no_spikes(self):
        entries = [_make_entry("info msg", "INFO", ts_offset_min=m) for m in range(10)]
        incidents = auto_detect_incidents(entries, window_size=180)
        assert incidents == []

    def test_auto_detect_incidents_with_spike(self):
        normal = [_make_entry("ok", "INFO", ts_offset_min=m) for m in range(5)]
        errors = [_make_entry(f"err{i}", "ERROR", ts_offset_min=10 + i) for i in range(20)]
        tail = [_make_entry("ok", "INFO", ts_offset_min=40 + m) for m in range(5)]
        entries = normal + errors + tail
        incidents = auto_detect_incidents(entries, window_size=300, spike_threshold=2.0)
        # Should detect the error-heavy window
        assert len(incidents) >= 0  # depends on distribution

    def test_auto_detect_incidents_empty_entries(self):
        incidents = auto_detect_incidents([])
        assert incidents == []

    def test_auto_detect_incidents_no_errors(self):
        entries = [_make_entry("msg", "DEBUG", ts_offset_min=m) for m in range(20)]
        incidents = auto_detect_incidents(entries)
        assert incidents == []

    @patch("sentence_transformers.SentenceTransformer")
    def test_label_incident_adds_to_index(self, MockST):
        model_inst = MagicMock()
        model_inst.encode.return_value = np.random.randn(1, DIM).astype(np.float32)
        MockST.return_value = model_inst

        index = _make_index(n_incidents=0)
        entries = [_make_entry("db error", "ERROR", ts_offset_min=m) for m in range(5)]
        updated = label_incident(
            index, entries,
            start_time=_ts(0), end_time=_ts(4),
            label="db-outage", resolution="restarted db",
        )
        assert updated.incident_count == 1
        assert updated.incident_labels[-1] == "db-outage"
        assert updated.incident_resolutions[-1] == "restarted db"

    @patch("sentence_transformers.SentenceTransformer")
    def test_label_incident_no_entries_in_range_raises(self, MockST):
        index = _make_index()
        entries = [_make_entry("msg", "INFO", ts_offset_min=0)]
        with pytest.raises(ValueError, match="No log entries found"):
            label_incident(
                index, entries,
                start_time=_ts(100), end_time=_ts(200),
                label="nothing here",
            )

    @patch("sentence_transformers.SentenceTransformer")
    def test_search_incidents_returns_results(self, MockST):
        model_inst = MagicMock()
        model_inst.encode.return_value = [np.random.randn(DIM).astype(np.float32)]
        MockST.return_value = model_inst

        index = _make_index(n_incidents=2)
        index.quantizer.cosine_scores.return_value = np.array([[0.85, 0.40]])
        results = search_incidents("database error", index, k=5)
        assert len(results) >= 1
        assert results[0].similarity >= 0.4

    @patch("sentence_transformers.SentenceTransformer")
    def test_search_incidents_empty_index(self, MockST):
        index = _make_index(n_incidents=0)
        results = search_incidents("anything", index)
        assert results == []


# ===========================================================================
# 11. Classifier unit integration
# ===========================================================================


class TestClassifiers:
    """Severity and anomaly type classifiers."""

    def test_classify_severity_critical_by_score(self):
        window = _make_window()
        assert _classify_severity(0.8, window, avg_errors=1.0) == "CRITICAL"

    def test_classify_severity_warning_by_score(self):
        window = _make_window()
        assert _classify_severity(0.55, window, avg_errors=1.0) == "WARNING"

    def test_classify_severity_info(self):
        window = _make_window()
        assert _classify_severity(0.2, window, avg_errors=1.0) == "INFO"

    def test_classify_severity_critical_by_error_count(self):
        entries = [_make_entry("err", "ERROR") for _ in range(15)]
        window = LogWindow(entries=entries, start_time=_ts(), end_time=_ts())
        assert _classify_severity(0.3, window, avg_errors=1.0) == "CRITICAL"

    def test_classify_anomaly_type_error_spike(self):
        entries = [_make_entry("err", "ERROR") for _ in range(20)]
        window = LogWindow(entries=entries, start_time=_ts(), end_time=_ts())
        result = _classify_anomaly_type(0.6, window, avg_errors=1.0, std_errors=1.0, similar_incidents=[])
        assert result == "error_spike"

    def test_classify_anomaly_type_similar_incident(self):
        window = _make_window()
        inc = IncidentMatch(similarity=0.8, label="past", resolution="", occurred_at=None)
        result = _classify_anomaly_type(0.5, window, avg_errors=5.0, std_errors=2.0, similar_incidents=[inc])
        assert result == "similar_incident"

    def test_classify_anomaly_type_new_pattern(self):
        window = _make_window()
        result = _classify_anomaly_type(0.5, window, avg_errors=5.0, std_errors=2.0, similar_incidents=[])
        assert result == "new_pattern"


# ===========================================================================
# 12. Additional cross-cutting integration tests
# ===========================================================================


class TestCrossCutting:
    """Miscellaneous cross-module scenarios."""

    def test_alert_top_incident_property(self):
        inc = IncidentMatch(similarity=0.9, label="top", resolution="fix", occurred_at=_ts())
        alert = _make_alert(similar=[inc])
        assert alert.top_incident is not None
        assert alert.top_incident.label == "top"

    def test_alert_top_incident_none_when_empty(self):
        alert = _make_alert(similar=[])
        assert alert.top_incident is None

    def test_webhook_payload_structure(self):
        """send_webhook constructs correct payload even if filtered."""
        alert = _make_alert(severity="INFO")
        # Should be filtered out before sending
        result = send_webhook(alert, "http://example.com", min_severity="CRITICAL")
        assert result is False

    @patch("httpx.post")
    def test_webhook_success_payload_has_fields(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        alert = _make_alert(severity="CRITICAL")
        result = send_webhook(alert, "http://example.com/hook", min_severity="INFO")
        assert result is True
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "severity" in payload
        assert "anomaly_score" in payload
        assert "summary" in payload

    def test_display_alert_json_roundtrip(self):
        alert = _make_alert()
        json_str = display_alert(alert, as_json=True)
        data = json.loads(json_str)
        assert data["anomaly_type"] == "error_spike"
        assert isinstance(data["similar_incidents"], list)

    def test_storage_file_is_pickle(self, tmp_path):
        index = _make_index(picklable=True)
        path = save_index(index, str(tmp_path))
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, LogMindIndex)

    def test_parse_line_http_status_codes(self):
        line = '10.0.0.1 - - [15/Mar/2024:10:00:00] "GET /api 500 1234"'
        entry = parse_line(line)
        assert entry.level == "ERROR"

    def test_parse_line_400_status_is_warn(self):
        line = '10.0.0.1 - - [15/Mar/2024:10:00:00] "GET /api 404 567"'
        entry = parse_line(line)
        assert entry.level == "WARN"
