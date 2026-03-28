"""Terminal display formatting using rich."""

from __future__ import annotations

import json
from typing import List, Optional

from .models import AnomalyAlert, IncidentMatch, LogMindIndex


def _severity_color(severity: str) -> str:
    return {"CRITICAL": "red", "WARNING": "yellow", "INFO": "blue"}.get(severity, "white")


def _severity_icon(severity: str) -> str:
    return {"CRITICAL": "[!!!]", "WARNING": "[!!]", "INFO": "[i]"}.get(severity, "[?]")


def display_alert(alert: AnomalyAlert, as_json: bool = False) -> str:
    """Format an anomaly alert for terminal display."""
    if as_json:
        data = {
            "severity": alert.severity,
            "anomaly_score": round(alert.anomaly_score, 4),
            "anomaly_type": alert.anomaly_type,
            "summary": alert.summary,
            "error_count": alert.current_window.error_count,
            "warn_count": alert.current_window.warn_count,
            "total_count": alert.current_window.total_count,
            "similar_incidents": [
                {
                    "label": inc.label,
                    "similarity": round(inc.similarity, 4),
                    "resolution": inc.resolution,
                    "occurred_at": str(inc.occurred_at) if inc.occurred_at else None,
                }
                for inc in alert.similar_incidents
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    lines = []
    icon = _severity_icon(alert.severity)

    lines.append("=" * 60)
    lines.append(f"  {icon} LogMind Alert: {alert.severity}")
    lines.append(f"  Anomaly Score: {alert.anomaly_score:.2f}")
    lines.append(f"  Type: {alert.anomaly_type}")
    if alert.current_window.start_time:
        lines.append(f"  Time: {alert.current_window.start_time}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(alert.summary)

    if alert.similar_incidents:
        lines.append("")
        lines.append("Similar Past Incidents:")
        for inc in alert.similar_incidents:
            lines.append(
                f"  [{inc.similarity:.0%}] {inc.label}"
            )
            if inc.resolution:
                lines.append(f"       Resolution: {inc.resolution}")
            if inc.occurred_at:
                lines.append(f"       When: {inc.occurred_at}")

    lines.append("-" * 60)
    return "\n".join(lines)


def display_scan_report(
    alerts: List[AnomalyAlert],
    total_windows: int,
    as_json: bool = False,
) -> str:
    """Format batch scan report."""
    if as_json:
        data = {
            "total_windows": total_windows,
            "anomalies_found": len(alerts),
            "alerts": [
                {
                    "severity": a.severity,
                    "anomaly_score": round(a.anomaly_score, 4),
                    "anomaly_type": a.anomaly_type,
                    "summary": a.summary,
                    "time": str(a.current_window.start_time)
                    if a.current_window.start_time else None,
                }
                for a in alerts
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    lines = []
    lines.append("=" * 60)
    lines.append("  LogMind Scan Report")
    lines.append(f"  Windows analyzed: {total_windows}")
    lines.append(f"  Anomalies found: {len(alerts)}")
    lines.append("=" * 60)

    if not alerts:
        lines.append("")
        lines.append("[OK] No anomalies detected.")
    else:
        critical = sum(1 for a in alerts if a.severity == "CRITICAL")
        warning = sum(1 for a in alerts if a.severity == "WARNING")
        info = sum(1 for a in alerts if a.severity == "INFO")

        if critical:
            lines.append(f"  CRITICAL: {critical}")
        if warning:
            lines.append(f"  WARNING:  {warning}")
        if info:
            lines.append(f"  INFO:     {info}")

        lines.append("")
        for alert in alerts:
            lines.append(display_alert(alert))
            lines.append("")

    return "\n".join(lines)


def display_stats(index: LogMindIndex) -> str:
    """Format index statistics."""
    lines = []
    lines.append("=" * 60)
    lines.append("  LogMind Index Statistics")
    lines.append("=" * 60)
    lines.append(f"  Total log lines:      {index.total_lines:,}")
    lines.append(f"  Total windows:        {index.total_windows:,}")
    lines.append(f"  Labeled incidents:    {index.incident_count}")
    lines.append(f"  Error count:          {index.error_count:,}")
    lines.append(f"  Warning count:        {index.warn_count:,}")
    lines.append(f"  Avg errors/window:    {index.avg_errors_per_window:.1f}")
    lines.append(f"  Embedding model:      {index.model_name}")
    lines.append(f"  Embedding dim:        {index.embedding_dim}")
    lines.append(f"  Learn time:           {index.learn_time:.1f}s")
    lines.append(f"  Sources ({len(index.sources)}):")
    for s in index.sources[:10]:
        lines.append(f"    - {s}")
    if len(index.sources) > 10:
        lines.append(f"    ... and {len(index.sources) - 10} more")

    if index.incident_count > 0:
        lines.append("")
        lines.append(f"  Labeled Incidents ({index.incident_count}):")
        for i, label in enumerate(index.incident_labels):
            res = index.incident_resolutions[i] if i < len(index.incident_resolutions) else ""
            lines.append(f"    [{i+1}] {label}")
            if res:
                lines.append(f"        Resolution: {res}")

    lines.append("=" * 60)
    return "\n".join(lines)


def display_search_results(results: List[IncidentMatch]) -> str:
    """Format incident search results."""
    if not results:
        return "No matching incidents found."

    lines = []
    lines.append("=" * 60)
    lines.append(f"  LogMind Search Results ({len(results)} found)")
    lines.append("=" * 60)

    for i, inc in enumerate(results, 1):
        lines.append(f"\n  [{i}] {inc.similarity:.0%} match - {inc.label}")
        if inc.resolution:
            lines.append(f"      Resolution: {inc.resolution}")
        if inc.occurred_at:
            lines.append(f"      When: {inc.occurred_at}")

    lines.append("")
    return "\n".join(lines)
