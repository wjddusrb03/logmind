"""Log format auto-detection and parsing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import LogEntry

# ---------- timestamp patterns ----------

_TS_PATTERNS = [
    # ISO 8601: 2024-03-15T10:30:45.123Z
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\d.]*Z?", "%Y-%m-%dT%H:%M:%S"),
    # Common: 2024-03-15 10:30:45,123
    (r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.\d]*", "%Y-%m-%d %H:%M:%S"),
    # Syslog: Mar 15 10:30:45
    (r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", "%b %d %H:%M:%S"),
    # Nginx: 15/Mar/2024:10:30:45
    (r"\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}", "%d/%b/%Y:%H:%M:%S"),
    # Unix epoch (seconds)
    (r"^\d{10}(?:\.\d+)?$", "epoch"),
]

# ---------- level patterns ----------

_LEVEL_MAP = {
    "FATAL": "FATAL",
    "CRITICAL": "FATAL",
    "CRIT": "FATAL",
    "ERROR": "ERROR",
    "ERR": "ERROR",
    "WARN": "WARN",
    "WARNING": "WARN",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
    "TRACE": "DEBUG",
    "VERBOSE": "DEBUG",
    "NOTICE": "INFO",
}

_LEVEL_RE = re.compile(
    r"\b(" + "|".join(_LEVEL_MAP.keys()) + r")\b", re.IGNORECASE
)

# ---------- format-specific patterns ----------

_FORMATS: Dict[str, re.Pattern] = {
    "python": re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+[\d:,]+)\s+-\s+(\S+)\s+-\s+(\w+)\s+-\s+(.*)"
    ),
    "spring": re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+[\d:.]+)\s+(\w+)\s+\d+\s+---\s+"
        r"\[.*?\]\s+(\S+)\s+:\s+(.*)"
    ),
    "syslog": re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+(\S+?)(?:\[\d+\])?:\s+(.*)"
    ),
    "nginx_access": re.compile(
        r"(\S+)\s+-\s+-\s+\[(.+?)\]\s+\"(\w+)\s+(\S+)\s+\S+\"\s+(\d+)\s+(\d+)"
    ),
    "docker": re.compile(
        r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+(stdout|stderr)\s+(.*)"
    ),
    "simple": re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+[\d:,.]+)\s+(\w+)\s+(.*)"
    ),
}


def _parse_timestamp(text: str) -> Optional[datetime]:
    """Try to extract a timestamp from text."""
    for pattern, fmt in _TS_PATTERNS:
        m = re.search(pattern, text)
        if m:
            ts_str = m.group(0)
            if fmt == "epoch":
                try:
                    return datetime.fromtimestamp(float(ts_str))
                except (ValueError, OSError):
                    continue
            # Clean up: replace comma with dot for fractional seconds, remove Z
            clean = ts_str.replace(",", ".").replace("Z", "").strip()
            # Truncate microseconds if too long
            clean = re.sub(r"\.(\d{6})\d+", r".\1", clean)
            for try_fmt in [fmt, fmt + ".%f", fmt + ",%f"]:
                try:
                    return datetime.strptime(clean[:26], try_fmt)
                except ValueError:
                    continue
    return None


def _parse_level(text: str) -> str:
    """Extract log level from text."""
    m = _LEVEL_RE.search(text)
    if m:
        return _LEVEL_MAP.get(m.group(1).upper(), "UNKNOWN")
    # HTTP status codes
    status_m = re.search(r'\s([45]\d{2})\s', text)
    if status_m:
        code = int(status_m.group(1))
        if code >= 500:
            return "ERROR"
        if code >= 400:
            return "WARN"
    return "INFO"


def _parse_source(text: str) -> str:
    """Extract source/service name from text."""
    # Python logger: name - LEVEL -
    m = re.search(r"-\s+(\S+)\s+-\s+\w+\s+-", text)
    if m:
        return m.group(1)
    # Simple format: TIMESTAMP LEVEL  SourceName: message
    m = re.search(r"(?:INFO|ERROR|WARN|WARNING|DEBUG|FATAL|CRITICAL)\s+(\w+):", text)
    if m:
        return m.group(1)
    # Spring: [thread] c.m.ClassName :
    m = re.search(r"\]\s+(\S+)\s+:", text)
    if m:
        return m.group(1).split(".")[-1]
    # Syslog: hostname service:
    m = re.search(r"^\w+\s+\d+\s+[\d:]+\s+\S+\s+(\S+?)(?:\[|:)", text)
    if m:
        return m.group(1)
    return "unknown"


def _try_json(line: str) -> Optional[LogEntry]:
    """Try parsing as JSON log."""
    try:
        data = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    # Extract timestamp
    ts = None
    for key in ("timestamp", "time", "ts", "@timestamp", "datetime", "date"):
        if key in data:
            ts = _parse_timestamp(str(data[key]))
            if ts:
                break

    # Extract level
    level = "INFO"
    for key in ("level", "severity", "loglevel", "log_level", "lvl"):
        if key in data:
            raw_level = str(data[key]).upper()
            level = _LEVEL_MAP.get(raw_level, "INFO")
            break

    # Extract message
    message = ""
    for key in ("message", "msg", "text", "log", "body"):
        if key in data:
            message = str(data[key])
            break

    # Extract source
    source = "unknown"
    for key in ("source", "logger", "service", "name", "component", "module"):
        if key in data:
            source = str(data[key])
            break

    if not message:
        message = line.strip()

    return LogEntry(
        timestamp=ts,
        level=level,
        source=source,
        message=message,
        raw=line.strip(),
        metadata={k: str(v) for k, v in data.items()
                  if k not in ("timestamp", "time", "ts", "level", "message",
                               "msg", "source", "logger")},
    )


def auto_detect_format(lines: List[str]) -> str:
    """Detect log format from sample lines."""
    if not lines:
        return "simple"

    # Try JSON first
    json_count = 0
    for line in lines[:10]:
        if _try_json(line) is not None:
            json_count += 1
    if json_count >= len(lines[:10]) * 0.5:
        return "json"

    # Try each named format
    for fmt_name, pattern in _FORMATS.items():
        match_count = sum(1 for l in lines[:10] if pattern.match(l.strip()))
        if match_count >= len(lines[:10]) * 0.5:
            return fmt_name

    return "simple"


def parse_line(line: str, fmt: str = "auto") -> LogEntry:
    """Parse a single log line."""
    line = line.rstrip("\n\r")
    if not line.strip():
        return LogEntry(
            timestamp=None, level="DEBUG", source="unknown",
            message="", raw=line,
        )

    # JSON
    if fmt in ("json", "auto"):
        entry = _try_json(line)
        if entry:
            return entry

    # Named format
    if fmt in _FORMATS:
        m = _FORMATS[fmt].match(line.strip())
        if m:
            groups = m.groups()
            # Try first group for timestamp; if not found, search all groups
            ts = _parse_timestamp(groups[0]) if groups else None
            if ts is None:
                for g in groups[1:]:
                    ts = _parse_timestamp(g)
                    if ts:
                        break
            level = _parse_level(line)
            source = _parse_source(line)
            message = groups[-1] if groups else line
            return LogEntry(
                timestamp=ts, level=level, source=source,
                message=message, raw=line,
            )

    # Fallback: generic parsing
    ts = _parse_timestamp(line)
    level = _parse_level(line)
    source = _parse_source(line)

    # Remove timestamp from message
    message = line
    if ts:
        for pattern, _ in _TS_PATTERNS:
            message = re.sub(pattern, "", message, count=1).strip()
    # Remove level from message
    message = _LEVEL_RE.sub("", message, count=1).strip()
    # Clean up separators
    message = re.sub(r"^[\s\-:|\[\]]+", "", message).strip()

    return LogEntry(
        timestamp=ts, level=level, source=source,
        message=message if message else line, raw=line,
    )


def parse_lines(lines: List[str], fmt: str = "auto") -> List[LogEntry]:
    """Parse multiple log lines with multiline support."""
    if fmt == "auto" and lines:
        fmt = auto_detect_format(lines)

    entries: List[LogEntry] = []
    pending_multiline: List[str] = []

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Check if this line starts a new log entry (has timestamp or level)
        has_ts = _parse_timestamp(line) is not None
        has_level = _LEVEL_RE.search(line) is not None
        is_json = line.strip().startswith("{")
        is_new_entry = has_ts or is_json or (has_level and not line.startswith(" "))

        if is_new_entry:
            # Flush pending multiline
            if pending_multiline and entries:
                entries[-1].message += "\n" + "\n".join(pending_multiline)
                entries[-1].raw += "\n" + "\n".join(pending_multiline)
            pending_multiline = []

            entry = parse_line(line, fmt)
            entry.line_number = i + 1
            entries.append(entry)
        else:
            # Continuation line (stack trace, etc.)
            pending_multiline.append(line.rstrip())

    # Flush remaining
    if pending_multiline and entries:
        entries[-1].message += "\n" + "\n".join(pending_multiline)
        entries[-1].raw += "\n" + "\n".join(pending_multiline)

    return entries


def parse_file(path: str, fmt: str = "auto") -> List[LogEntry]:
    """Parse a log file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return parse_lines(lines, fmt)
