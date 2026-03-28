# LogMind - AI-Powered Log Anomaly Detection

> [한국어 문서](README_KO.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1%2C243%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LogMind** is a CLI tool that learns your normal log patterns and detects anomalies using semantic embeddings. When an anomaly is detected, it finds the most similar past incidents and suggests resolutions.

## Why LogMind?

```
grep "ERROR":    "Here are 500 lines containing ERROR"         (keyword match)
LogMind:         "This pattern is 91% similar to last month's  (semantic match)
                  DB outage. Resolution: increase pool_size"
```

Traditional log monitoring relies on keyword matching or static rules. LogMind takes a fundamentally different approach:

| Feature | grep / ELK Rules | LogMind |
|---|---|---|
| Detection method | Keyword / regex | Semantic embedding similarity |
| New patterns | Missed (no rule) | Detected automatically |
| Past incident matching | Manual search | Automatic similarity search |
| Resolution suggestions | None | From past incident labels |
| Compression | None | 3-bit TurboQuant (90%+ savings) |

## Architecture

```
Log Files ──> Parser ──> Embedder ──> Detector ──> Alerter
                │            │           │            │
           Auto-detect   Sentence    Anomaly      Slack/Discord
           6+ formats   Transformer  Scoring      Webhook
                │            │           │
           LogEntry     TurboQuant   Past Incident
           LogWindow    Compression  Matching
```

### Core Components

| Module | Description |
|---|---|
| `parser.py` | Auto-detects and parses 7 log formats (JSON, Python, Spring Boot, Syslog, Nginx, Docker, Simple) |
| `embedder.py` | Converts log windows to semantic vectors using `all-MiniLM-L6-v2` (384-dim), compresses with TurboQuant 3-bit |
| `detector.py` | Scores anomalies by comparing against learned normal patterns, classifies severity and type |
| `incidents.py` | Labels past incidents, auto-detects error spikes, semantic search across incidents |
| `alerter.py` | Sends alerts to Slack (Block Kit), Discord (Embeds), or generic webhooks |
| `collector.py` | Collects logs from files (glob), stdin, Docker containers, shell commands |
| `storage.py` | Persists/loads the learned index (pickle-based) |
| `display.py` | Rich terminal output with severity-colored alerts and formatted reports |
| `cli.py` | Click-based CLI with 6 commands |

## Installation

### Requirements
- Python 3.10+
- ~500MB disk space (for sentence-transformers model, downloaded once)

### Install from source

```bash
git clone https://github.com/wjddusrb03/logmind.git
cd logmind
pip install -e .
```

### Optional dependencies

```bash
# Slack/Discord/Webhook alerts
pip install -e ".[alerts]"

# Docker log collection
pip install -e ".[docker]"

# All optional dependencies
pip install -e ".[alerts,docker]"
```

## Quick Start

```bash
# 1. Learn normal patterns from your logs
logmind learn /var/log/app.log

# 2. Scan a log file for anomalies
logmind scan /var/log/app-today.log

# 3. Watch logs in real-time
tail -f /var/log/app.log | logmind watch -
```

## Detailed Usage

### 1. `logmind learn` - Learn Normal Patterns

Learns what "normal" looks like by building a vector index from your log files.

```bash
# Basic usage
logmind learn app.log

# Multiple files with glob
logmind learn /var/log/app/*.log

# Specify log format (auto-detected by default)
logmind learn --format python app.log
logmind learn --format json structured.log
logmind learn --format syslog /var/log/syslog
logmind learn --format nginx_access /var/log/nginx/access.log
logmind learn --format docker container.log
logmind learn --format spring boot.log

# Custom window size (default: 60 seconds)
logmind learn --window 120 app.log

# Custom embedding model
logmind learn --model all-MiniLM-L6-v2 app.log

# Custom quantization bits (default: 3)
logmind learn --bits 4 app.log

# Auto-detect and label error spike incidents
logmind learn --auto-detect-incidents app.log
```

**Output:**
```
LogMind Learn Complete
  Total lines:    15,234
  Windows:        254
  Error count:    43
  Warn count:     128
  Sources:        api, database, auth, scheduler
  Model:          all-MiniLM-L6-v2
  Learn time:     12.3s
  Index saved to: .logmind/index.pkl
```

### 2. `logmind scan` - Batch Anomaly Detection

Scans a log file and reports all detected anomalies.

```bash
# Basic scan
logmind scan new-errors.log

# High sensitivity (catches more anomalies)
logmind scan --sensitivity high app.log

# Low sensitivity (only critical anomalies)
logmind scan --sensitivity low app.log

# JSON output (for piping to other tools)
logmind scan --json app.log

# Specify window size
logmind scan --window 30 app.log
```

**Output:**
```
CRITICAL  Anomaly Score: 0.87  Type: similar_incident
  Summary: Error spike detected: 15 errors in window. Similar to past
           incident "DB pool exhausted" (91% match).
           Resolution: increase pool_size to 200

WARNING   Anomaly Score: 0.52  Type: new_pattern
  Summary: New pattern detected not seen during learning phase.
           3 errors from previously unseen source "payment-gateway".

--- Scan Report ---
  Total windows scanned: 45
  Anomalies detected:    2
  Critical: 1  Warning: 1  Info: 0
```

**Sensitivity levels:**

| Level | Anomaly Threshold | Error Spike Threshold | Use Case |
|---|---|---|---|
| `low` | 0.55 | 5.0x baseline | Production (reduce noise) |
| `medium` | 0.40 | 3.0x baseline | Default |
| `high` | 0.30 | 2.0x baseline | Investigation mode |

### 3. `logmind watch` - Real-Time Monitoring

Continuously monitors logs and alerts on anomalies in real-time.

```bash
# Watch a file (tail -f style)
logmind watch /var/log/app.log

# Watch from stdin (pipe)
tail -f /var/log/app.log | logmind watch -
kubectl logs -f my-pod | logmind watch -

# Watch with Slack alerts
logmind watch app.log --slack https://hooks.slack.com/services/T.../B.../xxx

# Watch with Discord alerts
logmind watch app.log --discord https://discord.com/api/webhooks/123/abc

# Watch Docker container logs
logmind watch --docker my-container

# Watch output of a command
logmind watch --command "journalctl -f -u myapp"

# Combine options
logmind watch app.log \
  --sensitivity high \
  --slack https://hooks.slack.com/services/xxx \
  --json
```

### 4. `logmind label` - Label Past Incidents

Labels a time range in your logs as an incident. This teaches LogMind to recognize similar patterns in the future and suggest resolutions.

```bash
# Label an incident with time range
logmind label "2024-03-15 10:00" "2024-03-15 11:00" app.log \
  --label "DB pool exhausted" \
  --resolution "Increased pool_size from 50 to 200"

# Label multiple log sources
logmind label "2024-03-15 10:00" "2024-03-15 11:00" \
  app.log db.log worker.log \
  --label "Cascading failure from DB" \
  --resolution "Added circuit breaker, increased timeouts"
```

**How it works:**
1. Parses the log files
2. Extracts entries within the time range
3. Creates a labeled window with the incident description and resolution
4. Embeds the window and adds it to the incident index
5. Future scans will match similar patterns and suggest the resolution

### 5. `logmind search` - Semantic Incident Search

Searches past labeled incidents using semantic similarity (not just keywords).

```bash
# Search for similar incidents
logmind search "database connection timeout"
logmind search "memory leak in worker process"
logmind search "high latency on API endpoints"

# Return more results
logmind search "disk full" -k 10
```

**Output:**
```
Search Results for: "database connection timeout"

  1. DB pool exhausted (92% match)
     Resolution: Increased pool_size from 50 to 200
     Occurred: 2024-03-15 10:00

  2. Redis connection storm (67% match)
     Resolution: Added connection pooling with max_connections=100
     Occurred: 2024-02-28 14:30
```

### 6. `logmind stats` - Index Statistics

Shows detailed statistics about the learned index.

```bash
logmind stats
```

**Output:**
```
LogMind Index Statistics
  Model:              all-MiniLM-L6-v2
  Embedding dim:      384
  Total lines:        15,234
  Total windows:      254
  Incident count:     3
  Error count:        43
  Warn count:         128
  Sources:            api, database, auth, scheduler, worker
  Avg errors/window:  0.17
  Learn time:         12.3s
```

## Supported Log Formats

LogMind auto-detects log formats. Supported formats:

### JSON Logs
```json
{"timestamp":"2024-03-15T10:30:45Z","level":"ERROR","message":"DB connection failed","source":"api"}
```

### Python Logging
```
2024-03-15 10:30:45,123 - myapp.database - ERROR - Connection pool exhausted
```

### Spring Boot
```
2024-03-15 10:30:45.123 ERROR 12345 --- [http-nio-8080-exec-1] c.m.app.DbService : Query timeout
```

### Syslog
```
Mar 15 10:30:45 webserver sshd[12345]: Failed password for root from 192.168.1.1
```

### Nginx Access Log
```
192.168.1.1 - - [15/Mar/2024:10:30:45 +0000] "GET /api/health HTTP/1.1" 500 42
```

### Docker Logs
```
2024-03-15T10:30:45.123456Z stderr ERROR: database system is not yet accepting connections
```

### Simple Format
```
2024-03-15 10:30:45 ERROR Something went wrong
```

## How It Works

### Learning Phase
1. **Parse**: Auto-detect format, extract timestamp/level/source/message
2. **Window**: Group entries into time-based windows (default 60s)
3. **Embed**: Convert each window to a 384-dim vector using `all-MiniLM-L6-v2`
4. **Compress**: Quantize vectors to 3-bit using TurboQuant (90%+ memory savings)
5. **Save**: Persist the index to `.logmind/index.pkl`

### Detection Phase
1. **Parse & Window**: Same as learning phase for new logs
2. **Embed**: Convert each window to a vector
3. **Score**: Compare against normal patterns using cosine similarity
4. **Classify**: Determine anomaly type and severity
5. **Match**: Find similar past incidents from labeled data
6. **Alert**: Display results or send to Slack/Discord/webhook

### Anomaly Types

| Type | Description | Trigger |
|---|---|---|
| `error_spike` | Sudden increase in error rate | Error count exceeds baseline by threshold |
| `new_pattern` | Pattern not seen during learning | Low similarity to all normal patterns |
| `similar_incident` | Matches a past labeled incident | High similarity to incident vectors |

### Severity Levels

| Severity | Criteria |
|---|---|
| `CRITICAL` | Score >= 0.7 OR error count >= 10 |
| `WARNING` | Score >= 0.5 OR error count >= 3 |
| `INFO` | Below WARNING thresholds |

## Integration Examples

### Slack Alerts

```bash
# Set up Slack webhook:
# 1. Go to https://api.slack.com/apps → Create New App
# 2. Enable Incoming Webhooks → Add New Webhook to Workspace
# 3. Copy the webhook URL

logmind watch app.log --slack https://hooks.slack.com/services/T.../B.../xxx
```

Slack messages use Block Kit format with:
- Color-coded severity header
- Anomaly score, type, error count fields
- Summary with context
- Similar past incidents with resolution suggestions

### Discord Alerts

```bash
# Set up Discord webhook:
# 1. Server Settings → Integrations → Webhooks → New Webhook
# 2. Copy the webhook URL

logmind watch app.log --discord https://discord.com/api/webhooks/123/abc
```

### Generic Webhook

```bash
# Any HTTP endpoint that accepts JSON POST
logmind watch app.log --webhook https://your-api.com/alerts
```

Webhook payload:
```json
{
  "severity": "CRITICAL",
  "anomaly_score": 0.87,
  "anomaly_type": "similar_incident",
  "summary": "Error spike detected...",
  "error_count": 15,
  "total_count": 42,
  "similar_incidents": [
    {"label": "DB pool exhausted", "similarity": 0.91, "resolution": "increase pool_size"}
  ]
}
```

### Kubernetes / Docker

```bash
# Monitor a Docker container
logmind watch --docker my-container --slack https://hooks.slack.com/...

# Monitor Kubernetes pod logs
kubectl logs -f deployment/my-app | logmind watch - --sensitivity high

# Monitor journalctl
logmind watch --command "journalctl -f -u myapp.service"
```

### CI/CD Pipeline

```bash
# In your CI pipeline, scan test logs for anomalies
logmind scan test-output.log --json > anomalies.json

# Fail the pipeline if critical anomalies found
logmind scan test-output.log --sensitivity high --json | \
  python -c "import json,sys; d=json.load(sys.stdin); sys.exit(1 if d.get('critical',0)>0 else 0)"
```

## Project Structure

```
logmind/
├── src/logmind/
│   ├── __init__.py          # Version info
│   ├── models.py            # Data models (LogEntry, LogWindow, AnomalyAlert, etc.)
│   ├── parser.py            # Log format auto-detection and parsing
│   ├── embedder.py          # Sentence embedding and TurboQuant compression
│   ├── detector.py          # Anomaly detection engine
│   ├── incidents.py         # Incident labeling and search
│   ├── alerter.py           # Slack/Discord/Webhook alert delivery
│   ├── collector.py         # Log source collectors (file, stdin, docker, command)
│   ├── storage.py           # Index persistence (save/load)
│   ├── display.py           # Terminal display formatting
│   └── cli.py               # Click CLI commands
├── tests/                   # 1,243 test cases across 24 files
│   ├── test_models.py       # Core data model tests
│   ├── test_parser.py       # Parser unit tests
│   ├── test_parser_stress.py        # Parser stress tests (121 cases)
│   ├── test_parser_advanced.py      # Advanced parser tests (95 cases)
│   ├── test_embedder.py     # Embedder tests
│   ├── test_detector.py     # Detector unit tests
│   ├── test_detector_stress.py      # Detector stress tests (73 cases)
│   ├── test_detector_boundary.py    # Boundary condition tests (48 cases)
│   ├── test_alerter.py      # Alerter unit tests
│   ├── test_alerter_advanced.py     # Alerter advanced tests (68 cases)
│   ├── test_collector.py    # Collector tests
│   ├── test_collector_stress.py     # Collector stress tests
│   ├── test_collector_advanced.py   # Collector advanced tests
│   ├── test_display.py      # Display tests
│   ├── test_display_stress.py       # Display stress tests
│   ├── test_display_rigor.py        # Display rigorous tests
│   ├── test_storage.py      # Storage tests
│   ├── test_storage_rigor.py        # Storage rigorous tests
│   ├── test_models_rigor.py         # Models exhaustive tests
│   ├── test_cli.py          # CLI unit tests
│   ├── test_cli_e2e.py      # CLI end-to-end tests
│   ├── test_incidents_stress.py     # Incident stress tests
│   ├── test_real_model.py           # Real AI model integration tests
│   ├── test_e2e_pipeline.py         # Full pipeline E2E tests
│   ├── test_integration_comprehensive.py  # Cross-module integration (103 cases)
│   ├── test_robustness.py           # Security & robustness tests (74 cases)
│   └── sample.log           # Sample log file for testing
├── pyproject.toml           # Build configuration
├── README.md                # English documentation
└── README_KO.md             # Korean documentation
```

## Testing

```bash
# Run fast tests (~22 seconds)
pytest tests/ --ignore=tests/test_real_model.py --ignore=tests/test_e2e_pipeline.py \
  --ignore=tests/test_cli_e2e.py --ignore=tests/test_incidents_stress.py

# Run all tests including AI model tests (~15 minutes)
pytest tests/

# Run specific test categories
pytest tests/test_parser.py tests/test_parser_stress.py tests/test_parser_advanced.py -v
pytest tests/test_robustness.py -v
pytest tests/test_integration_comprehensive.py -v
```

**Test Results:** 1,243 tests across 24 files, **0 failures**.

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Semantic embeddings (`all-MiniLM-L6-v2`, 384-dim) |
| `langchain-turboquant` | 3-bit vector compression (90%+ memory savings) |
| `click` | CLI framework |
| `rich` | Terminal formatting |
| `numpy` | Numerical operations |
| `httpx` (optional) | HTTP alerts (Slack/Discord/Webhook) |
| `docker` (optional) | Docker container log collection |

## License

MIT
