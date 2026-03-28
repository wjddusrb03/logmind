"""Incident management: labeling and searching past incidents."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np

from .models import IncidentMatch, LogEntry, LogMindIndex, LogWindow


def label_incident(
    index: LogMindIndex,
    entries: List[LogEntry],
    start_time: datetime,
    end_time: datetime,
    label: str,
    resolution: str = "",
    model_name: Optional[str] = None,
) -> LogMindIndex:
    """Label a time range as an incident and add to the index."""
    from sentence_transformers import SentenceTransformer

    model_name = model_name or index.model_name

    # Filter entries in the time range
    incident_entries = [
        e for e in entries
        if e.timestamp is not None
        and start_time <= e.timestamp <= end_time
    ]

    if not incident_entries:
        raise ValueError(
            f"No log entries found between {start_time} and {end_time}"
        )

    # Create incident window
    window = LogWindow(
        entries=incident_entries,
        start_time=start_time,
        end_time=end_time,
        label=label,
        resolution=resolution,
    )

    # Embed
    model = SentenceTransformer(model_name)
    text = window.to_embedding_text()
    vec = model.encode([text], show_progress_bar=False)
    vec = np.array(vec, dtype=np.float32)

    # Add to incident index
    if index.incident_compressed is not None:
        # Re-quantize with existing + new
        existing_vecs = index.quantizer.dequantize(index.incident_compressed)
        all_vecs = np.vstack([existing_vecs, vec])
    else:
        all_vecs = vec

    new_compressed = index.quantizer.quantize(all_vecs)

    # Update index
    index.incident_compressed = new_compressed
    index.incident_windows.append(window)
    index.incident_labels.append(label)
    index.incident_resolutions.append(resolution)
    index.incident_count = len(index.incident_windows)

    return index


def auto_detect_incidents(
    entries: List[LogEntry],
    window_size: int = 60,
    spike_threshold: float = 3.0,
) -> List[LogWindow]:
    """Auto-detect potential incident windows by error spikes."""
    from .embedder import _create_windows

    windows = _create_windows(entries, window_size)
    if not windows:
        return []

    # Compute error counts per window
    error_counts = [w.error_count for w in windows]
    if not error_counts or max(error_counts) == 0:
        return []

    avg = np.mean(error_counts)
    std = np.std(error_counts)
    threshold = avg + spike_threshold * max(std, 1.0)

    incidents = []
    for w in windows:
        if w.error_count > threshold:
            w.label = f"auto: {w.error_count} errors (threshold: {threshold:.0f})"
            incidents.append(w)

    return incidents


def search_incidents(
    query: str,
    index: LogMindIndex,
    k: int = 5,
    model_name: Optional[str] = None,
) -> List[IncidentMatch]:
    """Semantic search over past incidents."""
    if index.incident_compressed is None or not index.incident_windows:
        return []

    from sentence_transformers import SentenceTransformer

    model_name = model_name or index.model_name
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], show_progress_bar=False)[0]
    query_vec = np.array(query_vec, dtype=np.float32)

    scores = index.quantizer.cosine_scores(query_vec, index.incident_compressed)
    score_array = np.array(scores, dtype=np.float64).flatten()

    top_k = min(k, len(score_array))
    top_indices = np.argsort(score_array)[::-1][:top_k]

    results = []
    for idx in top_indices:
        idx = int(idx)
        sim = float(score_array[idx])
        if sim < 0.1:
            continue

        window = index.incident_windows[idx] if idx < len(index.incident_windows) else None
        label = index.incident_labels[idx] if idx < len(index.incident_labels) else ""
        resolution = index.incident_resolutions[idx] if idx < len(index.incident_resolutions) else ""

        results.append(IncidentMatch(
            similarity=sim,
            label=label,
            resolution=resolution,
            occurred_at=window.start_time if window else None,
            window=window,
        ))

    return results
