"""Anomaly detection engine."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .models import AnomalyAlert, IncidentMatch, LogMindIndex, LogWindow


# Sensitivity presets: (anomaly_threshold, error_spike_multiplier)
_SENSITIVITY = {
    "low": (0.55, 5.0),
    "medium": (0.40, 3.0),
    "high": (0.30, 2.0),
}


def compute_anomaly_score(
    window_vec: np.ndarray,
    index: LogMindIndex,
) -> float:
    """Compute anomaly score: how far current window is from normal patterns."""
    scores = index.quantizer.cosine_scores(window_vec, index.normal_compressed)
    score_array = np.array(scores, dtype=np.float64).flatten()

    if len(score_array) == 0:
        return 1.0

    # Max similarity to any normal pattern
    max_sim = float(np.max(score_array))
    # Anomaly = 1 - max_similarity (farther from normal = more anomalous)
    return max(0.0, 1.0 - max_sim)


def find_similar_incidents(
    window_vec: np.ndarray,
    index: LogMindIndex,
    k: int = 3,
) -> List[IncidentMatch]:
    """Find past incidents similar to current window."""
    if index.incident_compressed is None or not index.incident_windows:
        return []

    scores = index.quantizer.cosine_scores(
        window_vec, index.incident_compressed
    )
    score_array = np.array(scores, dtype=np.float64).flatten()

    # Get top-k
    if len(score_array) == 0:
        return []

    top_k = min(k, len(score_array))
    top_indices = np.argsort(score_array)[::-1][:top_k]

    matches = []
    for idx in top_indices:
        idx = int(idx)
        sim = float(score_array[idx])
        if sim < 0.3:  # too low to be meaningful
            continue

        window = index.incident_windows[idx] if idx < len(index.incident_windows) else None
        label = index.incident_labels[idx] if idx < len(index.incident_labels) else ""
        resolution = index.incident_resolutions[idx] if idx < len(index.incident_resolutions) else ""

        matches.append(IncidentMatch(
            similarity=sim,
            label=label,
            resolution=resolution,
            occurred_at=window.start_time if window else None,
            window=window,
        ))

    return matches


def _classify_severity(
    anomaly_score: float,
    window: LogWindow,
    avg_errors: float,
) -> str:
    """Classify alert severity."""
    if anomaly_score >= 0.7 or window.error_count >= max(10, avg_errors * 5):
        return "CRITICAL"
    if anomaly_score >= 0.5 or window.error_count >= max(3, avg_errors * 3):
        return "WARNING"
    return "INFO"


def _classify_anomaly_type(
    anomaly_score: float,
    window: LogWindow,
    avg_errors: float,
    std_errors: float,
    similar_incidents: List[IncidentMatch],
) -> str:
    """Classify what kind of anomaly this is."""
    # Error spike: sudden increase in error rate
    error_threshold = avg_errors + max(3.0 * std_errors, 3.0)
    if window.error_count > error_threshold:
        return "error_spike"

    # Similar to past incident
    if similar_incidents and similar_incidents[0].similarity >= 0.6:
        return "similar_incident"

    # New unknown pattern
    return "new_pattern"


def _generate_summary(
    anomaly_type: str,
    window: LogWindow,
    similar: List[IncidentMatch],
) -> str:
    """Generate human-readable alert summary."""
    if anomaly_type == "error_spike":
        msg = (
            f"Error spike detected: {window.error_count} errors "
            f"in {window.window_size}s window"
        )
    elif anomaly_type == "similar_incident" and similar:
        top = similar[0]
        msg = (
            f"Pattern similar to past incident: \"{top.label}\" "
            f"({top.similarity:.0%} match)"
        )
    else:
        msg = "Unusual log pattern detected (not seen in normal baseline)"

    # Add top error messages
    errors = [e for e in window.entries if e.is_error]
    if errors:
        unique_errors = []
        seen = set()
        for e in errors:
            short = e.message[:80]
            if short not in seen:
                seen.add(short)
                unique_errors.append(e)
            if len(unique_errors) >= 3:
                break
        if unique_errors:
            msg += "\nTop errors:"
            for e in unique_errors:
                msg += f"\n  [{e.source}] {e.message[:120]}"

    return msg


def detect(
    window: LogWindow,
    window_vec: np.ndarray,
    index: LogMindIndex,
    sensitivity: str = "medium",
) -> Optional[AnomalyAlert]:
    """Detect anomaly in a log window.

    Returns AnomalyAlert if anomalous, None otherwise.
    """
    threshold, _ = _SENSITIVITY.get(sensitivity, _SENSITIVITY["medium"])

    # Compute anomaly score
    anomaly_score = compute_anomaly_score(window_vec, index)

    # Check threshold
    if anomaly_score < threshold:
        return None

    # Find similar past incidents
    similar = find_similar_incidents(window_vec, index)

    # Classify
    anomaly_type = _classify_anomaly_type(
        anomaly_score, window, index.avg_errors_per_window,
        index.std_errors_per_window, similar,
    )
    severity = _classify_severity(
        anomaly_score, window, index.avg_errors_per_window,
    )
    summary = _generate_summary(anomaly_type, window, similar)

    return AnomalyAlert(
        current_window=window,
        anomaly_score=anomaly_score,
        anomaly_type=anomaly_type,
        similar_incidents=similar,
        severity=severity,
        summary=summary,
    )
