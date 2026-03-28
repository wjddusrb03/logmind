# LogMind - AI 로그 이상 탐지

> [English README](README.md)

**LogMind**는 정상 로그 패턴을 학습하고, 과거 장애와 유사한 패턴이 감지되면 자동으로 경고합니다.

## 왜 LogMind인가?

```
grep "ERROR":   "ERROR가 포함된 500줄입니다"                    (키워드 매칭)
LogMind:        "이 패턴은 지난달 DB 장애와 91% 유사합니다.      (의미 매칭)
                 해결 방법: pool_size 증가"
```

## 빠른 시작

```bash
# 설치
git clone https://github.com/wjddusrb03/logmind.git
cd logmind && pip install -e .

# 1. 정상 패턴 학습
logmind learn app.log

# 2. 과거 장애 라벨링
logmind label "2024-03-15 10:00" "2024-03-15 11:00" app.log \
  --label "DB 커넥션 풀 고갈" --resolution "pool_size 50→200 증가"

# 3. 이상 탐지 (일괄)
logmind scan new-errors.log

# 4. 실시간 감시
tail -f /var/log/app.log | logmind watch -

# 5. 장애 검색
logmind search "메모리 누수"
```

## 전체 명령어

| 명령어 | 설명 |
|---|---|
| `logmind learn 파일` | 정상 로그 패턴 학습 |
| `logmind scan 파일` | 로그 파일 이상 탐지 (일괄) |
| `logmind watch 파일` | 실시간 로그 감시 |
| `logmind label 시작 끝 파일` | 장애 시점 라벨링 |
| `logmind search "쿼리"` | 과거 장애 의미 검색 |
| `logmind stats` | 인덱스 통계 |

## 라이선스

MIT
