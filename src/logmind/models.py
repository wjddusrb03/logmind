"""Data models for LogMind."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """A single parsed log line."""

    timestamp: Optional[datetime]
    level: str  # DEBUG, INFO, WARN, ERROR, FATAL, UNKNOWN
    source: str  # service/file name
    message: str  # log message body
    raw: str  # original line
    metadata: Dict[str, str] = field(default_factory=dict)
    line_number: int = 0

    @property
    def is_error(self) -> bool:
        return self.level in ("ERROR", "FATAL", "CRITICAL")

    @property
    def is_warn(self) -> bool:
        return self.level in ("WARN", "WARNING")


@dataclass
class LogWindow:
    """A time-based window of log entries."""

    entries: List[LogEntry]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    window_size: int = 60  # seconds
    label: str = ""  # incident label (empty = normal)
    resolution: str = ""  # how it was resolved

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if e.is_error)

    @property
    def warn_count(self) -> int:
        return sum(1 for e in self.entries if e.is_warn)

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def source_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.entries:
            dist[e.source] = dist.get(e.source, 0) + 1
        return dist

    @property
    def is_incident(self) -> bool:
        return bool(self.label)

    def to_embedding_text(self) -> str:
        """Convert window to text for embedding."""
        parts = []
        parts.append(
            f"[errors={self.error_count} warns={self.warn_count} "
            f"total={self.total_count}]"
        )

        sources = self.source_distribution
        if sources:
            top = sorted(sources.items(), key=lambda x: -x[1])[:5]
            parts.append("Sources: " + ", ".join(f"{s}({c})" for s, c in top))

        # Include error/warn/fatal messages (most informative)
        important = [
            e for e in self.entries
            if e.level in ("ERROR", "FATAL", "CRITICAL", "WARN", "WARNING")
        ]
        # Deduplicate similar messages
        seen = set()
        for entry in important[:50]:  # cap at 50
            key = entry.message[:100]
            if key not in seen:
                seen.add(key)
                parts.append(f"[{entry.level}] {entry.source}: {entry.message}")

        return "\n".join(parts)


@dataclass
class IncidentMatch:
    """A match to a past incident."""

    similarity: float
    label: str
    resolution: str
    occurred_at: Optional[datetime]
    window: Optional[LogWindow] = None


@dataclass
class AnomalyAlert:
    """An anomaly detection alert."""

    current_window: LogWindow
    anomaly_score: float  # 0~1 (1 = very anomalous)
    anomaly_type: str  # "error_spike", "new_pattern", "similar_incident"
    similar_incidents: List[IncidentMatch]
    severity: str  # "CRITICAL", "WARNING", "INFO"
    summary: str

    @property
    def top_incident(self) -> Optional[IncidentMatch]:
        if self.similar_incidents:
            return self.similar_incidents[0]
        return None


@dataclass
class LogMindIndex:
    """Persisted LogMind index."""

    # Normal pattern vectors (compressed)
    normal_compressed: Any  # CompressedVectors
    normal_windows: List[LogWindow]

    # Incident pattern vectors (compressed)
    incident_compressed: Optional[Any]  # CompressedVectors or None
    incident_windows: List[LogWindow]
    incident_labels: List[str]
    incident_resolutions: List[str]

    # Quantizer
    quantizer: Any  # TurboQuantizer

    # Model info
    model_name: str
    embedding_dim: int

    # Stats
    total_lines: int
    total_windows: int
    incident_count: int
    error_count: int
    warn_count: int
    sources: List[str]
    learn_time: float

    # Baseline stats for anomaly detection
    avg_errors_per_window: float = 0.0
    std_errors_per_window: float = 0.0
    avg_warns_per_window: float = 0.0
