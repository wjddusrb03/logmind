"""Tests for alerter formatting."""

import pytest
from datetime import datetime
from logmind.models import AnomalyAlert, IncidentMatch, LogEntry, LogWindow
from logmind.alerter import _format_slack_message, _format_discord_message


def _make_alert():
    entries = [LogEntry(None, "ERROR", "DB", "fail", "raw")]
    window = LogWindow(entries, datetime(2024,1,1), datetime(2024,1,1))
    inc = IncidentMatch(0.9, "DB down", "restart", datetime(2024,1,1))
    return AnomalyAlert(window, 0.8, "similar_incident", [inc], "CRITICAL", "Test")


class TestSlackFormat:
    def test_has_blocks(self):
        msg = _format_slack_message(_make_alert())
        assert "blocks" in msg
        assert len(msg["blocks"]) >= 3

    def test_contains_severity(self):
        msg = _format_slack_message(_make_alert())
        header_text = msg["blocks"][0]["text"]["text"]
        assert "CRITICAL" in header_text


class TestDiscordFormat:
    def test_has_embeds(self):
        msg = _format_discord_message(_make_alert())
        assert "embeds" in msg
        assert len(msg["embeds"]) == 1

    def test_color_red_for_critical(self):
        msg = _format_discord_message(_make_alert())
        assert msg["embeds"][0]["color"] == 0xFF0000

    def test_contains_fields(self):
        msg = _format_discord_message(_make_alert())
        fields = msg["embeds"][0]["fields"]
        assert len(fields) >= 3
        names = [f["name"] for f in fields]
        assert "Anomaly Score" in names
