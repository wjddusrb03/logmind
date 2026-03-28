# LogMind - AI-Powered Log Anomaly Detection

> [한국어 문서](README_KO.md)

**LogMind** learns your normal log patterns and detects anomalies by comparing against past incidents using semantic embeddings.

## Why LogMind?

```
grep "ERROR":    "Here are 500 lines containing ERROR"         (keyword match)
LogMind:         "This pattern is 91% similar to last month's  (semantic match)
                  DB outage. Resolution: increase pool_size"
```

## Quick Start

```bash
# Install
git clone https://github.com/wjddusrb03/logmind.git
cd logmind && pip install -e .

# 1. Learn normal patterns
logmind learn app.log

# 2. Label past incidents
logmind label "2024-03-15 10:00" "2024-03-15 11:00" app.log \
  --label "DB pool exhausted" --resolution "increase pool_size"

# 3. Scan for anomalies
logmind scan new-errors.log

# 4. Watch in real-time
tail -f /var/log/app.log | logmind watch -

# 5. Search past incidents
logmind search "memory leak"
```

## License

MIT
