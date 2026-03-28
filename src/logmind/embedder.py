"""Log embedding and TurboQuant compression."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .models import LogEntry, LogMindIndex, LogWindow

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BITS = 3
DEFAULT_WINDOW_SIZE = 60  # seconds


def _create_windows(
    entries: List[LogEntry],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> List[LogWindow]:
    """Group log entries into time-based windows."""
    if not entries:
        return []

    # Separate entries with and without timestamps
    timed = [e for e in entries if e.timestamp is not None]
    untimed = [e for e in entries if e.timestamp is None]

    windows: List[LogWindow] = []

    if timed:
        timed.sort(key=lambda e: e.timestamp)  # type: ignore
        current_entries: List[LogEntry] = []
        window_start = timed[0].timestamp

        for entry in timed:
            assert entry.timestamp is not None
            assert window_start is not None
            elapsed = (entry.timestamp - window_start).total_seconds()

            if elapsed >= window_size and current_entries:
                windows.append(LogWindow(
                    entries=current_entries,
                    start_time=window_start,
                    end_time=current_entries[-1].timestamp,
                    window_size=window_size,
                ))
                current_entries = []
                window_start = entry.timestamp

            current_entries.append(entry)

        if current_entries:
            windows.append(LogWindow(
                entries=current_entries,
                start_time=window_start,
                end_time=current_entries[-1].timestamp,
                window_size=window_size,
            ))

    # Handle untimed entries: group by line count
    if untimed and not timed:
        chunk_size = max(10, len(untimed) // max(1, len(untimed) // 50))
        for i in range(0, len(untimed), chunk_size):
            chunk = untimed[i:i + chunk_size]
            windows.append(LogWindow(
                entries=chunk,
                start_time=None,
                end_time=None,
                window_size=window_size,
            ))

    return windows


def build_index(
    entries: List[LogEntry],
    model_name: str = DEFAULT_MODEL,
    bits: int = DEFAULT_BITS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    incident_windows: Optional[List[LogWindow]] = None,
) -> LogMindIndex:
    """Build a LogMind index from parsed log entries."""
    from langchain_turboquant import TurboQuantizer
    from sentence_transformers import SentenceTransformer

    start = time.time()

    # Create windows
    windows = _create_windows(entries, window_size)
    if not windows:
        raise ValueError("No log windows could be created. Check your log file.")

    # Load model
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Embed normal windows
    texts = [w.to_embedding_text() for w in windows]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Compress
    quantizer = TurboQuantizer(dim=embedding_dim, bits=bits)
    normal_compressed = quantizer.quantize(embeddings)

    # Compute baseline stats
    error_counts = [w.error_count for w in windows]
    warn_counts = [w.warn_count for w in windows]
    avg_errors = float(np.mean(error_counts)) if error_counts else 0.0
    std_errors = float(np.std(error_counts)) if error_counts else 0.0
    avg_warns = float(np.mean(warn_counts)) if warn_counts else 0.0

    # Handle incident windows
    incident_compressed = None
    incident_labels: List[str] = []
    incident_resolutions: List[str] = []
    inc_windows: List[LogWindow] = incident_windows or []

    if inc_windows:
        inc_texts = [w.to_embedding_text() for w in inc_windows]
        inc_embeddings = model.encode(inc_texts, show_progress_bar=False)
        inc_embeddings = np.array(inc_embeddings, dtype=np.float32)
        incident_compressed = quantizer.quantize(inc_embeddings)
        incident_labels = [w.label for w in inc_windows]
        incident_resolutions = [w.resolution for w in inc_windows]

    # Collect stats
    all_sources = set()
    total_errors = 0
    total_warns = 0
    for w in windows:
        all_sources.update(w.source_distribution.keys())
        total_errors += w.error_count
        total_warns += w.warn_count

    elapsed = time.time() - start

    return LogMindIndex(
        normal_compressed=normal_compressed,
        normal_windows=windows,
        incident_compressed=incident_compressed,
        incident_windows=inc_windows,
        incident_labels=incident_labels,
        incident_resolutions=incident_resolutions,
        quantizer=quantizer,
        model_name=model_name,
        embedding_dim=embedding_dim,
        total_lines=len(entries),
        total_windows=len(windows),
        incident_count=len(inc_windows),
        error_count=total_errors,
        warn_count=total_warns,
        sources=sorted(all_sources),
        learn_time=elapsed,
        avg_errors_per_window=avg_errors,
        std_errors_per_window=std_errors,
        avg_warns_per_window=avg_warns,
    )


_model_cache: dict = {}


def _get_model(model_name: str = DEFAULT_MODEL):
    """Get or create a cached SentenceTransformer model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_window(
    window: LogWindow,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Embed a single window for real-time detection."""
    model = _get_model(model_name)
    text = window.to_embedding_text()
    vec = model.encode([text], show_progress_bar=False)[0]
    return np.array(vec, dtype=np.float32)
