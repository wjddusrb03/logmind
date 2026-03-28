"""Advanced tests for alerter formatting, filtering, and edge cases.

Covers Slack block structure, Discord embed details, min_severity filtering,
webhook payload structure, truncation, unicode, and boundary conditions.

Run with:  pytest tests/test_alerter_advanced.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from logmind.alerter import (
    _format_discord_message,
    _format_slack_message,
    send_discord,
    send_slack,
    send_webhook,
)
from logmind.models import AnomalyAlert, IncidentMatch, LogEntry, LogWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(error_count: int = 1, total: int = 5) -> LogWindow:
    entries = []
    for i in range(total):
        level = "ERROR" if i < error_count else "INFO"
        entries.append(LogEntry(datetime(2024, 1, 1), level, "svc", f"msg{i}", f"raw{i}"))
    return LogWindow(entries, datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1))


def _make_incident(similarity: float = 0.9, label: str = "DB crash",
                   resolution: str = "restart", ts: datetime | None = None) -> IncidentMatch:
    return IncidentMatch(similarity, label, resolution, ts or datetime(2024, 1, 1))


def _make_alert(
    severity: str = "CRITICAL",
    anomaly_type: str = "error_spike",
    score: float = 0.85,
    summary: str = "High error rate detected",
    incidents: list[IncidentMatch] | None = None,
    error_count: int = 3,
    total: int = 10,
) -> AnomalyAlert:
    window = _make_window(error_count, total)
    if incidents is None:
        incidents = [_make_incident()]
    return AnomalyAlert(window, score, anomaly_type, incidents, severity, summary)


# Module-level mock httpx used by send_* tests
_httpx_mock = MagicMock()


@pytest.fixture(autouse=False)
def mock_httpx_module():
    """Inject a mock httpx into sys.modules so lazy import picks it up."""
    _httpx_mock.reset_mock()
    resp = MagicMock()
    resp.status_code = 200
    _httpx_mock.post.return_value = resp
    old = sys.modules.get("httpx")
    sys.modules["httpx"] = _httpx_mock
    yield _httpx_mock
    if old is not None:
        sys.modules["httpx"] = old
    else:
        sys.modules.pop("httpx", None)


# ===================================================================
# 1. Slack message format - block types, field structure, emoji mapping
# ===================================================================

class TestSlackBlockTypes:
    """Verify Slack Block Kit structure in detail."""

    def test_header_block_type(self):
        msg = _format_slack_message(_make_alert())
        assert msg["blocks"][0]["type"] == "header"

    def test_header_text_type_plain(self):
        msg = _format_slack_message(_make_alert())
        assert msg["blocks"][0]["text"]["type"] == "plain_text"

    def test_section_fields_block_type(self):
        msg = _format_slack_message(_make_alert())
        assert msg["blocks"][1]["type"] == "section"

    def test_section_fields_count(self):
        msg = _format_slack_message(_make_alert())
        fields = msg["blocks"][1]["fields"]
        assert len(fields) == 4

    def test_field_types_are_mrkdwn(self):
        msg = _format_slack_message(_make_alert())
        for f in msg["blocks"][1]["fields"]:
            assert f["type"] == "mrkdwn"

    def test_anomaly_score_field_formatted(self):
        alert = _make_alert(score=0.7531)
        msg = _format_slack_message(alert)
        score_field = msg["blocks"][1]["fields"][0]
        assert "0.75" in score_field["text"]

    def test_type_field_shows_anomaly_type(self):
        alert = _make_alert(anomaly_type="new_pattern")
        msg = _format_slack_message(alert)
        type_field = msg["blocks"][1]["fields"][1]
        assert "new_pattern" in type_field["text"]

    def test_errors_field_shows_error_count(self):
        alert = _make_alert(error_count=7, total=20)
        msg = _format_slack_message(alert)
        err_field = msg["blocks"][1]["fields"][2]
        assert "7" in err_field["text"]

    def test_total_logs_field(self):
        alert = _make_alert(total=42)
        msg = _format_slack_message(alert)
        total_field = msg["blocks"][1]["fields"][3]
        assert "42" in total_field["text"]

    def test_summary_section_block(self):
        alert = _make_alert(summary="CPU spike on node-3")
        msg = _format_slack_message(alert)
        summary_block = msg["blocks"][2]
        assert summary_block["type"] == "section"
        assert "CPU spike on node-3" in summary_block["text"]["text"]


class TestSlackEmojiMapping:
    """Verify emoji for each severity level."""

    @pytest.mark.parametrize("severity,emoji", [
        ("CRITICAL", ":red_circle:"),
        ("WARNING", ":large_yellow_circle:"),
        ("INFO", ":large_blue_circle:"),
    ])
    def test_severity_emoji(self, severity, emoji):
        alert = _make_alert(severity=severity)
        msg = _format_slack_message(alert)
        header_text = msg["blocks"][0]["text"]["text"]
        assert emoji in header_text

    def test_unknown_severity_gets_white_circle(self):
        alert = _make_alert(severity="DEBUG")
        msg = _format_slack_message(alert)
        header_text = msg["blocks"][0]["text"]["text"]
        assert ":white_circle:" in header_text


class TestSlackIncidentSection:
    """Slack incident section details."""

    def test_incidents_add_extra_block(self):
        alert = _make_alert(incidents=[_make_incident()])
        msg = _format_slack_message(alert)
        assert len(msg["blocks"]) == 4  # header + fields + summary + incidents

    def test_no_incidents_no_extra_block(self):
        alert = _make_alert(incidents=[])
        msg = _format_slack_message(alert)
        assert len(msg["blocks"]) == 3

    def test_incident_label_in_text(self):
        inc = _make_incident(label="Redis timeout")
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        incident_block = msg["blocks"][3]
        assert "Redis timeout" in incident_block["text"]["text"]

    def test_incident_similarity_formatted_as_percent(self):
        inc = _make_incident(similarity=0.87)
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        incident_block = msg["blocks"][3]
        assert "87%" in incident_block["text"]["text"]

    def test_incident_resolution_shown(self):
        inc = _make_incident(resolution="Scale horizontally")
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        incident_block = msg["blocks"][3]
        assert "Scale horizontally" in incident_block["text"]["text"]

    def test_max_three_incidents_shown(self):
        incs = [_make_incident(label=f"inc_{i}") for i in range(5)]
        alert = _make_alert(incidents=incs)
        msg = _format_slack_message(alert)
        incident_text = msg["blocks"][3]["text"]["text"]
        assert "inc_0" in incident_text
        assert "inc_2" in incident_text
        assert "inc_4" not in incident_text  # only first 3

    def test_incident_without_resolution_omits_line(self):
        inc = _make_incident(resolution="")
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        incident_text = msg["blocks"][3]["text"]["text"]
        assert "Resolution:" not in incident_text


# ===================================================================
# 2. Discord embed details
# ===================================================================

class TestDiscordEmbedDetails:
    """Discord embed color codes, field names, description truncation."""

    @pytest.mark.parametrize("severity,color", [
        ("CRITICAL", 0xFF0000),
        ("WARNING", 0xFFAA00),
        ("INFO", 0x0099FF),
    ])
    def test_color_for_severity(self, severity, color):
        alert = _make_alert(severity=severity)
        msg = _format_discord_message(alert)
        assert msg["embeds"][0]["color"] == color

    def test_unknown_severity_grey_color(self):
        alert = _make_alert(severity="TRACE")
        msg = _format_discord_message(alert)
        assert msg["embeds"][0]["color"] == 0x808080

    def test_embed_title_contains_severity(self):
        alert = _make_alert(severity="WARNING")
        msg = _format_discord_message(alert)
        assert "WARNING" in msg["embeds"][0]["title"]

    def test_field_names_present(self):
        msg = _format_discord_message(_make_alert())
        names = [f["name"] for f in msg["embeds"][0]["fields"]]
        assert "Anomaly Score" in names
        assert "Type" in names
        assert "Errors" in names

    def test_inline_true_for_main_fields(self):
        msg = _format_discord_message(_make_alert())
        for f in msg["embeds"][0]["fields"][:3]:
            assert f["inline"] is True

    def test_description_truncated_at_2000(self):
        long_summary = "X" * 5000
        alert = _make_alert(summary=long_summary)
        msg = _format_discord_message(alert)
        assert len(msg["embeds"][0]["description"]) == 2000

    def test_description_not_truncated_when_short(self):
        alert = _make_alert(summary="short")
        msg = _format_discord_message(alert)
        assert msg["embeds"][0]["description"] == "short"

    def test_similar_incident_fields_inline_false(self):
        inc = _make_incident(label="OOM kill")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        for f in inc_fields:
            assert f["inline"] is False

    def test_max_two_incidents_in_fields(self):
        incs = [_make_incident(label=f"evt_{i}") for i in range(5)]
        alert = _make_alert(incidents=incs)
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert len(inc_fields) == 2

    def test_incident_no_resolution_fallback_text(self):
        inc = _make_incident(resolution="")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert inc_fields[0]["value"] == "No resolution recorded"


# ===================================================================
# 3. send_slack / send_discord / send_webhook - min_severity filtering
# ===================================================================

class TestMinSeverityFiltering:
    """Test min_severity filtering for all severity x min_severity combos."""

    @pytest.mark.parametrize("alert_sev,min_sev,expected", [
        # min_severity=INFO -> all pass
        ("INFO", "INFO", True),
        ("WARNING", "INFO", True),
        ("CRITICAL", "INFO", True),
        # min_severity=WARNING -> INFO blocked
        ("INFO", "WARNING", False),
        ("WARNING", "WARNING", True),
        ("CRITICAL", "WARNING", True),
        # min_severity=CRITICAL -> only CRITICAL passes
        ("INFO", "CRITICAL", False),
        ("WARNING", "CRITICAL", False),
        ("CRITICAL", "CRITICAL", True),
    ])
    def test_send_slack_severity_filter(self, mock_httpx_module, alert_sev, min_sev, expected):
        alert = _make_alert(severity=alert_sev)
        result = send_slack(alert, "https://hooks.slack.com/test", min_severity=min_sev)
        assert result is expected

    @pytest.mark.parametrize("alert_sev,min_sev,expected", [
        ("INFO", "INFO", True),
        ("WARNING", "INFO", True),
        ("CRITICAL", "INFO", True),
        ("INFO", "WARNING", False),
        ("WARNING", "WARNING", True),
        ("CRITICAL", "WARNING", True),
        ("INFO", "CRITICAL", False),
        ("WARNING", "CRITICAL", False),
        ("CRITICAL", "CRITICAL", True),
    ])
    def test_send_discord_severity_filter(self, mock_httpx_module, alert_sev, min_sev, expected):
        alert = _make_alert(severity=alert_sev)
        result = send_discord(alert, "https://discord.com/api/webhooks/test", min_severity=min_sev)
        assert result is expected

    @pytest.mark.parametrize("alert_sev,min_sev,expected", [
        ("INFO", "INFO", True),
        ("INFO", "WARNING", False),
        ("INFO", "CRITICAL", False),
        ("WARNING", "WARNING", True),
        ("CRITICAL", "WARNING", True),
        ("CRITICAL", "CRITICAL", True),
    ])
    def test_send_webhook_severity_filter(self, mock_httpx_module, alert_sev, min_sev, expected):
        alert = _make_alert(severity=alert_sev)
        result = send_webhook(alert, "https://example.com/webhook", min_severity=min_sev)
        assert result is expected


# ===================================================================
# 4. Webhook payload structure
# ===================================================================

class TestWebhookPayloadStructure:
    """Verify send_webhook constructs proper JSON payload."""

    def test_payload_has_all_required_fields(self, mock_httpx_module):
        alert = _make_alert()
        send_webhook(alert, "https://example.com/hook")

        call_kwargs = mock_httpx_module.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert "severity" in payload
        assert "anomaly_score" in payload
        assert "anomaly_type" in payload
        assert "summary" in payload
        assert "error_count" in payload
        assert "total_count" in payload
        assert "similar_incidents" in payload

    def test_payload_incident_fields(self, mock_httpx_module):
        inc = _make_incident(similarity=0.75, label="OOM", resolution="add RAM")
        alert = _make_alert(incidents=[inc])
        send_webhook(alert, "https://example.com/hook")

        payload = mock_httpx_module.post.call_args.kwargs.get("json") or mock_httpx_module.post.call_args[1]["json"]
        inc_data = payload["similar_incidents"][0]
        assert inc_data["label"] == "OOM"
        assert inc_data["similarity"] == 0.75
        assert inc_data["resolution"] == "add RAM"

    def test_payload_content_type_header(self, mock_httpx_module):
        send_webhook(_make_alert(), "https://example.com/hook")

        call_kwargs = mock_httpx_module.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Content-Type"] == "application/json"


# ===================================================================
# 5. Alert with 0, 1, 5 incidents - message formatting
# ===================================================================

class TestIncidentCountFormatting:
    """Formatting with different incident counts."""

    def test_zero_incidents_slack(self):
        alert = _make_alert(incidents=[])
        msg = _format_slack_message(alert)
        # 3 blocks: header, fields, summary (no incident section)
        assert len(msg["blocks"]) == 3

    def test_zero_incidents_discord(self):
        alert = _make_alert(incidents=[])
        msg = _format_discord_message(alert)
        # Only the 3 base fields
        assert len(msg["embeds"][0]["fields"]) == 3

    def test_one_incident_slack(self):
        alert = _make_alert(incidents=[_make_incident()])
        msg = _format_slack_message(alert)
        assert len(msg["blocks"]) == 4

    def test_one_incident_discord(self):
        alert = _make_alert(incidents=[_make_incident()])
        msg = _format_discord_message(alert)
        assert len(msg["embeds"][0]["fields"]) == 4  # 3 base + 1 incident

    def test_five_incidents_slack_caps_at_three(self):
        incs = [_make_incident(label=f"i{i}") for i in range(5)]
        alert = _make_alert(incidents=incs)
        msg = _format_slack_message(alert)
        text = msg["blocks"][3]["text"]["text"]
        # Count pushpin emojis - should be 3
        assert text.count(":pushpin:") == 3

    def test_five_incidents_discord_caps_at_two(self):
        incs = [_make_incident(label=f"i{i}") for i in range(5)]
        alert = _make_alert(incidents=incs)
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert len(inc_fields) == 2


# ===================================================================
# 6. Very long summary truncation
# ===================================================================

class TestLongSummaryTruncation:
    """Verify truncation for long summaries."""

    def test_slack_summary_truncated_at_500(self):
        long_text = "A" * 5000
        alert = _make_alert(summary=long_text)
        msg = _format_slack_message(alert)
        summary_text = msg["blocks"][2]["text"]["text"]
        # Summary section includes "*Summary:*\n" prefix
        assert len(summary_text) < 600

    def test_discord_summary_truncated_at_2000(self):
        long_text = "B" * 5000
        alert = _make_alert(summary=long_text)
        msg = _format_discord_message(alert)
        assert len(msg["embeds"][0]["description"]) == 2000

    def test_short_summary_not_truncated_slack(self):
        alert = _make_alert(summary="brief")
        msg = _format_slack_message(alert)
        assert "brief" in msg["blocks"][2]["text"]["text"]


# ===================================================================
# 7. Unicode in labels/summaries
# ===================================================================

class TestUnicodeContent:
    """Korean text, emoji, and special characters."""

    def test_korean_summary_slack(self):
        alert = _make_alert(summary="\ub370\uc774\ud130\ubca0\uc774\uc2a4 \uc5f0\uacb0 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4")
        msg = _format_slack_message(alert)
        assert "\ub370\uc774\ud130\ubca0\uc774\uc2a4" in msg["blocks"][2]["text"]["text"]

    def test_korean_summary_discord(self):
        alert = _make_alert(summary="\uc11c\ubc84 \uacfc\ubd80\ud558 \uacbd\uace0")
        msg = _format_discord_message(alert)
        assert "\uc11c\ubc84 \uacfc\ubd80\ud558" in msg["embeds"][0]["description"]

    def test_emoji_in_label(self):
        inc = _make_incident(label="Server crash \U0001f4a5")
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        assert "\U0001f4a5" in msg["blocks"][3]["text"]["text"]

    def test_korean_incident_label_discord(self):
        inc = _make_incident(label="\uba54\ubaa8\ub9ac \ubd80\uc871")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert "\uba54\ubaa8\ub9ac \ubd80\uc871" in inc_fields[0]["name"]


# ===================================================================
# 8. None timestamps, empty strings
# ===================================================================

class TestNoneAndEmptyValues:
    """Edge cases with None timestamps and empty strings."""

    def test_none_timestamps_in_window(self):
        entries = [LogEntry(None, "ERROR", "svc", "fail", "raw")]
        window = LogWindow(entries, None, None)
        alert = AnomalyAlert(window, 0.5, "error_spike", [], "WARNING", "test")
        msg = _format_slack_message(alert)
        assert "blocks" in msg

    def test_empty_summary(self):
        alert = _make_alert(summary="")
        msg = _format_slack_message(alert)
        # Should still produce valid blocks
        assert len(msg["blocks"]) >= 3

    def test_empty_anomaly_type(self):
        alert = _make_alert(anomaly_type="")
        msg = _format_discord_message(alert)
        assert "embeds" in msg

    def test_incident_none_occurred_at(self):
        inc = IncidentMatch(0.8, "test", "fix it", None)
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        assert "test" in msg["blocks"][3]["text"]["text"]

    def test_empty_label_incident(self):
        inc = _make_incident(label="")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        # Should not crash
        assert len(msg["embeds"][0]["fields"]) >= 3


# ===================================================================
# 9. Severity ordering
# ===================================================================

class TestSeverityOrdering:
    """Verify INFO < WARNING < CRITICAL ordering in filtering."""

    def test_info_blocked_by_warning(self, mock_httpx_module):
        alert = _make_alert(severity="INFO")
        result = send_slack(alert, "https://hooks.slack.com/test", min_severity="WARNING")
        assert result is False
        mock_httpx_module.post.assert_not_called()

    def test_critical_passes_all_filters(self, mock_httpx_module):
        alert = _make_alert(severity="CRITICAL")
        for min_sev in ("INFO", "WARNING", "CRITICAL"):
            mock_httpx_module.reset_mock()
            resp = MagicMock()
            resp.status_code = 200
            mock_httpx_module.post.return_value = resp
            assert send_slack(alert, "https://hooks.slack.com/test", min_severity=min_sev) is True

    def test_unknown_severity_treated_as_lowest(self, mock_httpx_module):
        alert = _make_alert(severity="TRACE")
        result = send_slack(alert, "https://hooks.slack.com/test", min_severity="WARNING")
        assert result is False

    def test_unknown_min_severity_treated_as_lowest(self, mock_httpx_module):
        alert = _make_alert(severity="INFO")
        result = send_slack(alert, "https://hooks.slack.com/test", min_severity="UNKNOWN")
        assert result is True


# ===================================================================
# 10. Multiple incidents with various similarity scores
# ===================================================================

class TestMultipleIncidentsSimilarity:
    """Multiple incidents with different similarity scores."""

    def test_similarity_formatted_correctly_slack(self):
        incs = [
            _make_incident(similarity=0.95, label="high_sim"),
            _make_incident(similarity=0.50, label="mid_sim"),
            _make_incident(similarity=0.10, label="low_sim"),
        ]
        alert = _make_alert(incidents=incs)
        msg = _format_slack_message(alert)
        text = msg["blocks"][3]["text"]["text"]
        assert "95%" in text
        assert "50%" in text
        assert "10%" in text

    def test_similarity_in_discord_field_name(self):
        inc = _make_incident(similarity=0.42, label="Network issue")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert "42%" in inc_fields[0]["name"]

    def test_zero_similarity_incident(self):
        inc = _make_incident(similarity=0.0, label="no_match")
        alert = _make_alert(incidents=[inc])
        msg = _format_slack_message(alert)
        text = msg["blocks"][3]["text"]["text"]
        assert "0%" in text

    def test_full_similarity_incident(self):
        inc = _make_incident(similarity=1.0, label="exact_match")
        alert = _make_alert(incidents=[inc])
        msg = _format_discord_message(alert)
        inc_fields = [f for f in msg["embeds"][0]["fields"] if "Similar:" in f["name"]]
        assert "100%" in inc_fields[0]["name"]


# ===================================================================
# 11. HTTP error handling
# ===================================================================

class TestHTTPErrorHandling:
    """Verify send functions handle HTTP errors gracefully."""

    def test_slack_returns_false_on_500(self, mock_httpx_module):
        mock_httpx_module.post.return_value.status_code = 500
        assert send_slack(_make_alert(), "https://hooks.slack.com/test") is False

    def test_discord_accepts_204(self, mock_httpx_module):
        mock_httpx_module.post.return_value.status_code = 204
        assert send_discord(_make_alert(), "https://discord.com/api/webhooks/test") is True

    def test_webhook_returns_false_on_400_plus(self, mock_httpx_module):
        mock_httpx_module.post.return_value.status_code = 400
        assert send_webhook(_make_alert(), "https://example.com/hook") is False

    def test_webhook_returns_true_on_201(self, mock_httpx_module):
        mock_httpx_module.post.return_value.status_code = 201
        assert send_webhook(_make_alert(), "https://example.com/hook") is True

    def test_slack_returns_false_on_import_error(self):
        """If httpx is not installed, send_slack returns False."""
        # Temporarily remove httpx from sys.modules to simulate missing package
        old = sys.modules.pop("httpx", None)
        # Also block future imports
        import builtins
        _real_import = builtins.__import__

        def _fail_httpx(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("no httpx")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _fail_httpx
        try:
            alert = _make_alert()
            result = send_slack(alert, "https://hooks.slack.com/test")
            assert result is False
        finally:
            builtins.__import__ = _real_import
            if old is not None:
                sys.modules["httpx"] = old

    def test_send_slack_network_exception(self, mock_httpx_module):
        mock_httpx_module.post.side_effect = ConnectionError("timeout")
        result = send_slack(_make_alert(), "https://hooks.slack.com/test")
        assert result is False

    def test_send_discord_network_exception(self, mock_httpx_module):
        mock_httpx_module.post.side_effect = TimeoutError("slow")
        result = send_discord(_make_alert(), "https://discord.com/api/webhooks/test")
        assert result is False
