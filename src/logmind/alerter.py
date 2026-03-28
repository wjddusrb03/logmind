"""Alert channels: Slack, Discord, Webhook."""

from __future__ import annotations

import json
from typing import Optional

from .models import AnomalyAlert


def _format_slack_message(alert: AnomalyAlert) -> dict:
    """Format alert as Slack Block Kit message."""
    severity_emoji = {
        "CRITICAL": ":red_circle:",
        "WARNING": ":large_yellow_circle:",
        "INFO": ":large_blue_circle:",
    }
    emoji = severity_emoji.get(alert.severity, ":white_circle:")

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} LogMind: {alert.severity} - {alert.anomaly_type}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Anomaly Score:*\n{alert.anomaly_score:.2f}"},
                {"type": "mrkdwn", "text": f"*Type:*\n{alert.anomaly_type}"},
                {"type": "mrkdwn", "text": f"*Errors:*\n{alert.current_window.error_count}"},
                {"type": "mrkdwn", "text": f"*Total Logs:*\n{alert.current_window.total_count}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{alert.summary[:500]}"},
        },
    ]

    # Add similar incidents
    if alert.similar_incidents:
        incident_text = "*Similar Past Incidents:*\n"
        for inc in alert.similar_incidents[:3]:
            incident_text += (
                f"  :pushpin: {inc.similarity:.0%} - {inc.label}\n"
            )
            if inc.resolution:
                incident_text += f"    Resolution: {inc.resolution}\n"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": incident_text},
        })

    return {"blocks": blocks}


def _format_discord_message(alert: AnomalyAlert) -> dict:
    """Format alert as Discord embed message."""
    colors = {"CRITICAL": 0xFF0000, "WARNING": 0xFFAA00, "INFO": 0x0099FF}
    color = colors.get(alert.severity, 0x808080)

    fields = [
        {"name": "Anomaly Score", "value": f"{alert.anomaly_score:.2f}", "inline": True},
        {"name": "Type", "value": alert.anomaly_type, "inline": True},
        {"name": "Errors", "value": str(alert.current_window.error_count), "inline": True},
    ]

    if alert.similar_incidents:
        for inc in alert.similar_incidents[:2]:
            fields.append({
                "name": f"Similar: {inc.label} ({inc.similarity:.0%})",
                "value": inc.resolution or "No resolution recorded",
                "inline": False,
            })

    return {
        "embeds": [{
            "title": f"LogMind: {alert.severity}",
            "description": alert.summary[:2000],
            "color": color,
            "fields": fields,
        }]
    }


def send_slack(
    alert: AnomalyAlert,
    webhook_url: str,
    min_severity: str = "WARNING",
) -> bool:
    """Send alert to Slack webhook."""
    severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    if severity_order.get(alert.severity, 0) < severity_order.get(min_severity, 0):
        return False

    try:
        import httpx
        payload = _format_slack_message(alert)
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_discord(
    alert: AnomalyAlert,
    webhook_url: str,
    min_severity: str = "WARNING",
) -> bool:
    """Send alert to Discord webhook."""
    severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    if severity_order.get(alert.severity, 0) < severity_order.get(min_severity, 0):
        return False

    try:
        import httpx
        payload = _format_discord_message(alert)
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def send_webhook(
    alert: AnomalyAlert,
    url: str,
    min_severity: str = "WARNING",
) -> bool:
    """Send alert to a generic webhook as JSON."""
    severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    if severity_order.get(alert.severity, 0) < severity_order.get(min_severity, 0):
        return False

    payload = {
        "severity": alert.severity,
        "anomaly_score": alert.anomaly_score,
        "anomaly_type": alert.anomaly_type,
        "summary": alert.summary,
        "error_count": alert.current_window.error_count,
        "total_count": alert.current_window.total_count,
        "similar_incidents": [
            {
                "label": inc.label,
                "similarity": inc.similarity,
                "resolution": inc.resolution,
            }
            for inc in alert.similar_incidents
        ],
    }

    try:
        import httpx
        resp = httpx.post(
            url, json=payload, timeout=10,
            headers={"Content-Type": "application/json"},
        )
        return resp.status_code < 400
    except Exception:
        return False
