# LogMind - AI 로그 이상 탐지

> [English README](README.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1%2C243%20passed-brightgreen.svg)](#테스트)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LogMind**는 정상 로그 패턴을 학습하고, 이상 패턴이 감지되면 과거 유사 장애를 찾아 해결 방법을 제안하는 AI 기반 CLI 도구입니다.

## 왜 LogMind인가?

```
grep "ERROR":   "ERROR가 포함된 500줄입니다"                    (키워드 매칭)
LogMind:        "이 패턴은 지난달 DB 장애와 91% 유사합니다.      (의미 매칭)
                 해결 방법: pool_size 50 -> 200 증가"
```

기존 로그 모니터링은 키워드 매칭이나 정적 규칙에 의존합니다. LogMind는 근본적으로 다른 접근 방식을 사용합니다:

| 기능 | grep / ELK 규칙 | LogMind |
|---|---|---|
| 탐지 방식 | 키워드 / 정규식 | 시맨틱 임베딩 유사도 |
| 새로운 패턴 | 놓침 (규칙 없음) | 자동 탐지 |
| 과거 장애 매칭 | 수동 검색 | 자동 유사도 검색 |
| 해결 방법 제안 | 없음 | 과거 라벨에서 자동 추천 |
| 압축 | 없음 | 3비트 TurboQuant (90%+ 절약) |

## 아키텍처

```
로그 파일 ──> 파서 ──> 임베더 ──> 탐지기 ──> 알림
               │         │         │         │
          자동 포맷    Sentence   이상    Slack/Discord
           감지      Transformer  점수     Webhook
               │         │         │
          LogEntry   TurboQuant  과거 장애
          LogWindow   3비트 압축   매칭
```

### 핵심 모듈

| 모듈 | 설명 |
|---|---|
| `parser.py` | 7가지 로그 포맷 자동 감지 및 파싱 (JSON, Python, Spring Boot, Syslog, Nginx, Docker, Simple) |
| `embedder.py` | `all-MiniLM-L6-v2` (384차원) 시맨틱 벡터 변환, TurboQuant 3비트 압축 |
| `detector.py` | 학습된 정상 패턴 대비 이상 점수 산출, 심각도/유형 분류 |
| `incidents.py` | 과거 장애 라벨링, 에러 스파이크 자동 감지, 시맨틱 검색 |
| `alerter.py` | Slack (Block Kit), Discord (Embed), 웹훅 알림 발송 |
| `collector.py` | 파일 (glob), stdin, Docker 컨테이너, 셸 명령어 로그 수집 |
| `storage.py` | 학습 인덱스 저장/로드 (pickle 기반) |
| `display.py` | 터미널 출력 포매팅 (심각도 색상 구분) |
| `cli.py` | Click 기반 CLI 6개 명령어 |

## 설치

### 요구 사항
- Python 3.10+
- 디스크 약 500MB (sentence-transformers 모델, 최초 1회 다운로드)

### 소스에서 설치

```bash
git clone https://github.com/wjddusrb03/logmind.git
cd logmind
pip install -e .
```

### 선택적 의존성

```bash
# Slack/Discord/웹훅 알림
pip install -e ".[alerts]"

# Docker 로그 수집
pip install -e ".[docker]"

# 모든 선택적 의존성
pip install -e ".[alerts,docker]"
```

## 빠른 시작

```bash
# 1. 정상 패턴 학습
logmind learn /var/log/app.log

# 2. 로그 파일 이상 탐지
logmind scan /var/log/app-today.log

# 3. 실시간 감시
tail -f /var/log/app.log | logmind watch -
```

## 상세 사용법

### 1. `logmind learn` - 정상 패턴 학습

로그 파일에서 "정상 상태"가 어떤 것인지 학습하여 벡터 인덱스를 구축합니다.

```bash
# 기본 사용
logmind learn app.log

# 여러 파일 (glob 패턴)
logmind learn /var/log/app/*.log

# 로그 포맷 지정 (기본: 자동 감지)
logmind learn --format python app.log
logmind learn --format json structured.log
logmind learn --format syslog /var/log/syslog
logmind learn --format nginx_access /var/log/nginx/access.log
logmind learn --format docker container.log
logmind learn --format spring boot.log

# 윈도우 크기 변경 (기본: 60초)
logmind learn --window 120 app.log

# 커스텀 임베딩 모델
logmind learn --model all-MiniLM-L6-v2 app.log

# 양자화 비트 수 변경 (기본: 3)
logmind learn --bits 4 app.log

# 에러 스파이크 자동 감지 & 라벨링
logmind learn --auto-detect-incidents app.log
```

**출력 예시:**
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

### 2. `logmind scan` - 일괄 이상 탐지

로그 파일을 스캔하여 모든 이상 패턴을 보고합니다.

```bash
# 기본 스캔
logmind scan new-errors.log

# 높은 민감도 (더 많은 이상 감지)
logmind scan --sensitivity high app.log

# 낮은 민감도 (심각한 이상만)
logmind scan --sensitivity low app.log

# JSON 출력 (다른 도구와 연동)
logmind scan --json app.log

# 윈도우 크기 지정
logmind scan --window 30 app.log
```

**출력 예시:**
```
CRITICAL  이상 점수: 0.87  유형: similar_incident
  요약: 에러 스파이크 감지: 윈도우 내 15개 에러.
        과거 장애 "DB 커넥션 풀 고갈"과 91% 유사.
        해결 방법: pool_size를 200으로 증가

WARNING   이상 점수: 0.52  유형: new_pattern
  요약: 학습 단계에서 본 적 없는 새로운 패턴 감지.
        이전에 없던 소스 "payment-gateway"에서 3개 에러.

--- 스캔 보고서 ---
  총 스캔 윈도우: 45
  이상 감지:      2
  Critical: 1  Warning: 1  Info: 0
```

**민감도 설정:**

| 레벨 | 이상 임계값 | 에러 스파이크 임계값 | 사용 시나리오 |
|---|---|---|---|
| `low` | 0.55 | 기준선 5.0배 | 운영 환경 (노이즈 감소) |
| `medium` | 0.40 | 기준선 3.0배 | 기본값 |
| `high` | 0.30 | 기준선 2.0배 | 조사/디버깅 모드 |

### 3. `logmind watch` - 실시간 모니터링

로그를 지속적으로 감시하며 이상 발견 시 실시간 알림합니다.

```bash
# 파일 감시 (tail -f 방식)
logmind watch /var/log/app.log

# stdin에서 읽기 (파이프)
tail -f /var/log/app.log | logmind watch -
kubectl logs -f my-pod | logmind watch -

# Slack 알림 연동
logmind watch app.log --slack https://hooks.slack.com/services/T.../B.../xxx

# Discord 알림 연동
logmind watch app.log --discord https://discord.com/api/webhooks/123/abc

# Docker 컨테이너 로그 감시
logmind watch --docker my-container

# 명령어 출력 감시
logmind watch --command "journalctl -f -u myapp"

# 옵션 조합
logmind watch app.log \
  --sensitivity high \
  --slack https://hooks.slack.com/services/xxx \
  --json
```

### 4. `logmind label` - 과거 장애 라벨링

로그의 특정 시간 범위를 장애로 라벨링합니다. LogMind가 미래에 유사한 패턴을 인식하고 해결 방법을 제안할 수 있게 됩니다.

```bash
# 시간 범위 지정 라벨링
logmind label "2024-03-15 10:00" "2024-03-15 11:00" app.log \
  --label "DB 커넥션 풀 고갈" \
  --resolution "pool_size를 50에서 200으로 증가"

# 여러 로그 파일에서 라벨링
logmind label "2024-03-15 10:00" "2024-03-15 11:00" \
  app.log db.log worker.log \
  --label "DB 장애로 인한 연쇄 실패" \
  --resolution "서킷 브레이커 추가, 타임아웃 증가"
```

**작동 원리:**
1. 로그 파일 파싱
2. 지정 시간 범위 내 항목 추출
3. 장애 설명과 해결 방법으로 라벨 생성
4. 윈도우를 임베딩하여 장애 인덱스에 추가
5. 이후 스캔 시 유사 패턴 매칭 및 해결 방법 제안

### 5. `logmind search` - 시맨틱 장애 검색

과거 라벨링된 장애를 시맨틱 유사도로 검색합니다 (키워드가 아닌 의미 기반).

```bash
# 유사 장애 검색
logmind search "데이터베이스 연결 타임아웃"
logmind search "워커 프로세스 메모리 누수"
logmind search "API 엔드포인트 높은 지연시간"

# 더 많은 결과 반환
logmind search "디스크 가득 참" -k 10
```

**출력 예시:**
```
검색 결과: "데이터베이스 연결 타임아웃"

  1. DB 커넥션 풀 고갈 (92% 일치)
     해결 방법: pool_size를 50에서 200으로 증가
     발생 시각: 2024-03-15 10:00

  2. Redis 연결 폭주 (67% 일치)
     해결 방법: max_connections=100 연결 풀링 추가
     발생 시각: 2024-02-28 14:30
```

### 6. `logmind stats` - 인덱스 통계

학습된 인덱스의 상세 통계를 표시합니다.

```bash
logmind stats
```

**출력 예시:**
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

## 지원 로그 포맷

LogMind는 로그 포맷을 자동 감지합니다. 지원 포맷:

### JSON 로그
```json
{"timestamp":"2024-03-15T10:30:45Z","level":"ERROR","message":"DB 연결 실패","source":"api"}
```

### Python 로깅
```
2024-03-15 10:30:45,123 - myapp.database - ERROR - 커넥션 풀 고갈
```

### Spring Boot
```
2024-03-15 10:30:45.123 ERROR 12345 --- [http-nio-8080-exec-1] c.m.app.DbService : 쿼리 타임아웃
```

### Syslog
```
Mar 15 10:30:45 webserver sshd[12345]: Failed password for root from 192.168.1.1
```

### Nginx 접근 로그
```
192.168.1.1 - - [15/Mar/2024:10:30:45 +0000] "GET /api/health HTTP/1.1" 500 42
```

### Docker 로그
```
2024-03-15T10:30:45.123456Z stderr ERROR: 데이터베이스 시스템 미준비
```

### 심플 포맷
```
2024-03-15 10:30:45 ERROR 오류 발생
```

## 작동 원리

### 학습 단계
1. **파싱**: 포맷 자동 감지, 타임스탬프/레벨/소스/메시지 추출
2. **윈도우**: 시간 기반 윈도우로 그룹화 (기본 60초)
3. **임베딩**: `all-MiniLM-L6-v2`로 각 윈도우를 384차원 벡터로 변환
4. **압축**: TurboQuant 3비트 양자화 (90%+ 메모리 절약)
5. **저장**: `.logmind/index.pkl`에 인덱스 저장

### 탐지 단계
1. **파싱 & 윈도우**: 새 로그에 학습 단계와 동일한 처리 적용
2. **임베딩**: 각 윈도우를 벡터로 변환
3. **점수 산출**: 정상 패턴과 코사인 유사도 비교
4. **분류**: 이상 유형 및 심각도 결정
5. **매칭**: 라벨링된 과거 장애에서 유사 사례 검색
6. **알림**: 결과 표시 또는 Slack/Discord/웹훅 발송

### 이상 유형

| 유형 | 설명 | 트리거 조건 |
|---|---|---|
| `error_spike` | 갑작스러운 에러 증가 | 에러 수가 기준선 대비 임계값 초과 |
| `new_pattern` | 학습 시 보지 못한 패턴 | 모든 정상 패턴과 유사도 낮음 |
| `similar_incident` | 과거 라벨링된 장애와 유사 | 장애 벡터와 유사도 높음 |

### 심각도 수준

| 심각도 | 기준 |
|---|---|
| `CRITICAL` | 점수 >= 0.7 또는 에러 10개 이상 |
| `WARNING` | 점수 >= 0.5 또는 에러 3개 이상 |
| `INFO` | WARNING 기준 미달 |

## 연동 예시

### Slack 알림

```bash
# Slack 웹훅 설정:
# 1. https://api.slack.com/apps -> 새 앱 만들기
# 2. Incoming Webhooks 활성화 -> 새 웹훅 추가
# 3. 웹훅 URL 복사

logmind watch app.log --slack https://hooks.slack.com/services/T.../B.../xxx
```

Slack 메시지에 포함되는 정보:
- 심각도별 색상 구분 헤더
- 이상 점수, 유형, 에러 수 필드
- 상황 요약
- 유사 과거 장애 및 해결 방법 제안

### Discord 알림

```bash
# Discord 웹훅 설정:
# 1. 서버 설정 -> 연동 -> 웹후크 -> 새 웹후크
# 2. 웹훅 URL 복사

logmind watch app.log --discord https://discord.com/api/webhooks/123/abc
```

### 범용 웹훅

```bash
# JSON POST를 받는 모든 HTTP 엔드포인트
logmind watch app.log --webhook https://your-api.com/alerts
```

웹훅 페이로드:
```json
{
  "severity": "CRITICAL",
  "anomaly_score": 0.87,
  "anomaly_type": "similar_incident",
  "summary": "에러 스파이크 감지...",
  "error_count": 15,
  "total_count": 42,
  "similar_incidents": [
    {"label": "DB 커넥션 풀 고갈", "similarity": 0.91, "resolution": "pool_size 증가"}
  ]
}
```

### Kubernetes / Docker

```bash
# Docker 컨테이너 모니터링
logmind watch --docker my-container --slack https://hooks.slack.com/...

# Kubernetes 파드 로그 모니터링
kubectl logs -f deployment/my-app | logmind watch - --sensitivity high

# journalctl 모니터링
logmind watch --command "journalctl -f -u myapp.service"
```

### CI/CD 파이프라인

```bash
# CI 파이프라인에서 테스트 로그 이상 탐지
logmind scan test-output.log --json > anomalies.json

# Critical 이상 발견 시 파이프라인 실패
logmind scan test-output.log --sensitivity high --json | \
  python -c "import json,sys; d=json.load(sys.stdin); sys.exit(1 if d.get('critical',0)>0 else 0)"
```

## 프로젝트 구조

```
logmind/
├── src/logmind/
│   ├── __init__.py          # 버전 정보
│   ├── models.py            # 데이터 모델 (LogEntry, LogWindow, AnomalyAlert 등)
│   ├── parser.py            # 로그 포맷 자동 감지 및 파싱
│   ├── embedder.py          # 문장 임베딩 및 TurboQuant 압축
│   ├── detector.py          # 이상 탐지 엔진
│   ├── incidents.py         # 장애 라벨링 및 검색
│   ├── alerter.py           # Slack/Discord/웹훅 알림 발송
│   ├── collector.py         # 로그 소스 수집기 (파일, stdin, docker, 명령어)
│   ├── storage.py           # 인덱스 영속화 (저장/로드)
│   ├── display.py           # 터미널 출력 포매팅
│   └── cli.py               # Click CLI 명령어
├── tests/                   # 24개 파일, 1,243개 테스트 케이스
├── pyproject.toml           # 빌드 설정
├── README.md                # 영문 문서
└── README_KO.md             # 한국어 문서
```

## 테스트

```bash
# 빠른 테스트 실행 (~22초)
pytest tests/ --ignore=tests/test_real_model.py --ignore=tests/test_e2e_pipeline.py \
  --ignore=tests/test_cli_e2e.py --ignore=tests/test_incidents_stress.py

# AI 모델 포함 전체 테스트 (~15분)
pytest tests/

# 특정 카테고리 실행
pytest tests/test_parser.py tests/test_parser_stress.py tests/test_parser_advanced.py -v
pytest tests/test_robustness.py -v
pytest tests/test_integration_comprehensive.py -v
```

**테스트 결과:** 24개 파일, 1,243개 테스트, **실패 0건**.

## 의존성

| 패키지 | 용도 |
|---|---|
| `sentence-transformers` | 시맨틱 임베딩 (`all-MiniLM-L6-v2`, 384차원) |
| `langchain-turboquant` | 3비트 벡터 압축 (90%+ 메모리 절약) |
| `click` | CLI 프레임워크 |
| `rich` | 터미널 포매팅 |
| `numpy` | 수치 연산 |
| `httpx` (선택) | HTTP 알림 (Slack/Discord/웹훅) |
| `docker` (선택) | Docker 컨테이너 로그 수집 |

## 라이선스

MIT
