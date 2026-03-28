"""LogMind CLI - AI-powered log anomaly detection."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Optional

import click

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="logmind")
def main():
    """LogMind - AI-powered log anomaly detection.

    Learn normal log patterns, detect anomalies, and find similar past incidents.
    """
    pass


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--format", "fmt", default="auto",
              help="Log format: auto, json, python, spring, syslog, nginx, docker, simple")
@click.option("--window", default=60, help="Window size in seconds (default: 60)")
@click.option("--model", default="all-MiniLM-L6-v2", help="Embedding model")
@click.option("--bits", default=3, type=click.IntRange(2, 4), help="Compression bits")
@click.option("--auto-detect-incidents", "auto_incidents", is_flag=True,
              help="Auto-detect incident windows from error spikes")
def learn(paths, fmt, window, model, bits, auto_incidents):
    """Learn normal log patterns from log file(s).

    Examples:

      logmind learn app.log

      logmind learn /var/log/*.log --window 120

      logmind learn error.log --auto-detect-incidents
    """
    from .collector import FileCollector
    from .embedder import build_index
    from .incidents import auto_detect_incidents
    from .parser import parse_lines
    from .storage import save_index

    click.echo(f"Reading log files: {', '.join(paths)}")

    try:
        collector = FileCollector(list(paths))
    except FileNotFoundError as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

    raw_lines = collector.read_all()
    click.echo(f"Loaded {len(raw_lines):,} lines")

    if not raw_lines:
        click.echo("[ERROR] No log lines found.", err=True)
        sys.exit(1)

    entries = parse_lines(raw_lines, fmt)
    click.echo(f"Parsed {len(entries):,} log entries")

    # Auto-detect incidents if requested
    incident_windows = None
    if auto_incidents:
        incident_windows = auto_detect_incidents(entries, window)
        click.echo(f"Auto-detected {len(incident_windows)} incident windows")

    click.echo("Building index (embedding + compression)...")
    try:
        index = build_index(
            entries,
            model_name=model,
            bits=bits,
            window_size=window,
            incident_windows=incident_windows,
        )
    except ValueError as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

    path = save_index(index)
    click.echo(f"\n[OK] Index saved to {path}")
    click.echo(f"  Windows: {index.total_windows:,}")
    click.echo(f"  Errors: {index.error_count:,}  Warnings: {index.warn_count:,}")
    click.echo(f"  Sources: {len(index.sources)}")
    click.echo(f"  Avg errors/window: {index.avg_errors_per_window:.1f}")
    click.echo(f"  Learn time: {index.learn_time:.1f}s")


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--format", "fmt", default="auto", help="Log format")
@click.option("--window", default=60, help="Window size in seconds")
@click.option("--sensitivity", default="medium",
              type=click.Choice(["low", "medium", "high"]),
              help="Detection sensitivity")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def scan(paths, fmt, window, sensitivity, as_json):
    """Scan log file(s) for anomalies (batch mode).

    Examples:

      logmind scan error.log

      logmind scan /var/log/app-*.log --sensitivity high
    """
    from .collector import FileCollector
    from .detector import detect
    from .display import display_scan_report
    from .embedder import _create_windows, embed_window
    from .parser import parse_lines
    from .storage import load_index

    index = load_index()
    if index is None:
        click.echo("[ERROR] No index found. Run 'logmind learn' first.", err=True)
        sys.exit(1)

    try:
        collector = FileCollector(list(paths))
    except FileNotFoundError as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

    raw_lines = collector.read_all()
    entries = parse_lines(raw_lines, fmt)
    windows = _create_windows(entries, window)

    if not as_json:
        click.echo(f"Scanning {len(windows)} windows...")

    alerts = []
    for w in windows:
        vec = embed_window(w, index.model_name)
        alert = detect(w, vec, index, sensitivity)
        if alert:
            alerts.append(alert)

    report = display_scan_report(alerts, len(windows), as_json)
    click.echo(report)

    # Exit code for CI
    if any(a.severity == "CRITICAL" for a in alerts):
        sys.exit(2)
    if any(a.severity == "WARNING" for a in alerts):
        sys.exit(1)


@main.command()
@click.argument("paths", nargs=-1)
@click.option("--docker", "docker_container", default=None,
              help="Docker container name")
@click.option("--command", "cmd", default=None,
              help="Shell command to collect logs from")
@click.option("--format", "fmt", default="auto", help="Log format")
@click.option("--window", default=60, help="Window size in seconds")
@click.option("--sensitivity", default="medium",
              type=click.Choice(["low", "medium", "high"]),
              help="Detection sensitivity")
@click.option("--slack", default=None, help="Slack webhook URL")
@click.option("--discord", default=None, help="Discord webhook URL")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def watch(paths, docker_container, cmd, fmt, window, sensitivity,
          slack, discord, as_json):
    """Watch log source in real-time for anomalies.

    Examples:

      logmind watch app.log

      logmind watch --docker my-container

      logmind watch app.log --slack https://hooks.slack.com/...

      tail -f /var/log/syslog | logmind watch -
    """
    from .alerter import send_discord, send_slack
    from .detector import detect
    from .display import display_alert
    from .embedder import embed_window
    from .models import LogWindow
    from .parser import parse_line
    from .storage import load_index

    index = load_index()
    if index is None:
        click.echo("[ERROR] No index found. Run 'logmind learn' first.", err=True)
        sys.exit(1)

    # Select collector
    if docker_container:
        from .collector import DockerCollector
        collector = DockerCollector(docker_container)
        source_name = f"docker:{docker_container}"
    elif cmd:
        from .collector import CommandCollector
        collector = CommandCollector(cmd)
        source_name = f"cmd:{cmd[:30]}"
    elif paths and paths[0] == "-":
        from .collector import StdinCollector
        collector = StdinCollector()
        source_name = "stdin"
    elif paths:
        from .collector import FileCollector
        collector = FileCollector(list(paths), follow=True)
        source_name = ", ".join(paths)
    else:
        click.echo("[ERROR] Specify log file(s), --docker, --command, or pipe to stdin", err=True)
        sys.exit(1)

    if not as_json:
        click.echo(f"[LogMind] Watching: {source_name}")
        click.echo(f"[LogMind] Sensitivity: {sensitivity}, Window: {window}s")
        click.echo(f"[LogMind] Press Ctrl+C to stop")
        click.echo("-" * 60)

    # Sliding window
    current_entries = []
    window_start = None
    alert_count = 0

    try:
        for line in collector.stream():
            entry = parse_line(line.rstrip("\n\r"), fmt)

            if entry.timestamp:
                if window_start is None:
                    window_start = entry.timestamp

                elapsed = (entry.timestamp - window_start).total_seconds()

                if elapsed >= window:
                    # Process window
                    if current_entries:
                        log_window = LogWindow(
                            entries=current_entries,
                            start_time=window_start,
                            end_time=entry.timestamp,
                            window_size=window,
                        )
                        vec = embed_window(log_window, index.model_name)
                        alert = detect(log_window, vec, index, sensitivity)

                        if alert:
                            alert_count += 1
                            output = display_alert(alert, as_json)
                            click.echo(output)

                            if slack:
                                send_slack(alert, slack)
                            if discord:
                                send_discord(alert, discord)

                    current_entries = []
                    window_start = entry.timestamp

            current_entries.append(entry)

    except KeyboardInterrupt:
        if not as_json:
            click.echo(f"\n[LogMind] Stopped. Alerts: {alert_count}")


@main.command()
@click.argument("start_time")
@click.argument("end_time")
@click.argument("log_paths", nargs=-1, required=True)
@click.option("--label", "label_text", required=True, help="Incident label/description")
@click.option("--resolution", default="", help="How it was resolved")
@click.option("--format", "fmt", default="auto", help="Log format")
def label(start_time, end_time, log_paths, label_text, resolution, fmt):
    """Label a time range as a known incident.

    Examples:

      logmind label "2024-03-15 10:00" "2024-03-15 11:00" app.log \\
        --label "DB connection pool exhausted" \\
        --resolution "Increased pool_size from 50 to 200"
    """
    from .collector import FileCollector
    from .incidents import label_incident
    from .parser import parse_lines
    from .storage import load_index, save_index

    index = load_index()
    if index is None:
        click.echo("[ERROR] No index found. Run 'logmind learn' first.", err=True)
        sys.exit(1)

    # Parse time range
    for time_fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            start = datetime.strptime(start_time, time_fmt)
            end = datetime.strptime(end_time, time_fmt)
            break
        except ValueError:
            continue
    else:
        click.echo("[ERROR] Cannot parse time. Use format: YYYY-MM-DD HH:MM", err=True)
        sys.exit(1)

    # Read log entries
    collector = FileCollector(list(log_paths))
    raw_lines = collector.read_all()
    entries = parse_lines(raw_lines, fmt)

    try:
        index = label_incident(index, entries, start, end, label_text, resolution)
    except ValueError as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

    save_index(index)
    click.echo(f"[OK] Incident labeled: \"{label_text}\"")
    click.echo(f"  Time range: {start} ~ {end}")
    if resolution:
        click.echo(f"  Resolution: {resolution}")
    click.echo(f"  Total incidents: {index.incident_count}")


@main.command()
@click.argument("query")
@click.option("-k", default=5, help="Number of results")
def search(query, k):
    """Search past incidents by meaning.

    Examples:

      logmind search "memory leak"

      logmind search "database timeout" -k 10
    """
    from .display import display_search_results
    from .incidents import search_incidents
    from .storage import load_index

    index = load_index()
    if index is None:
        click.echo("[ERROR] No index found. Run 'logmind learn' first.", err=True)
        sys.exit(1)

    if index.incident_count == 0:
        click.echo("No incidents labeled yet. Use 'logmind label' to add incidents.")
        return

    results = search_incidents(query, index, k)
    output = display_search_results(results)
    click.echo(output)


@main.command()
def stats():
    """Show index statistics."""
    from .display import display_stats
    from .storage import load_index

    index = load_index()
    if index is None:
        click.echo("[ERROR] No index found. Run 'logmind learn' first.", err=True)
        sys.exit(1)

    output = display_stats(index)
    click.echo(output)


if __name__ == "__main__":
    main()
