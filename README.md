# Multi Supertrend Backtest Framework

Python 백테스트 프레임워크 - 듀얼 Supertrend 전략 + 역추세(Counter-Trend) 오버레이.

## 특징

- **메인 전략**: 듀얼 Supertrend (ST1 + ST2) + RSI 상태머신 필터 + BB 기울기 필터
- **역추세 (CT)**: RSI 극단값 + 연속봉 기반 평균회귀 진입, DCA 지원
- **다단계 TP**: 설정 가능한 수익률 단계별 부분 청산
- **소프트 SL**: 봉 종가 기준 스탑로스 (봉 내 SL 없음)
- **도구**: 파라미터 최적화, 멀티 심볼 스캔, CT 비교, 주중 필터 비교

## 최고 성과 설정

| 심볼 | TF | Factor | ATR | RSI long_block | RSI short_block | 거래수 | 승률 | 수익 | PF |
|------|----|--------|-----|----------------|-----------------|--------|------|------|----|
| PI/USDT:USDT | 12m | 5.4 | 10 | 80 | 25 | 455 | 56.9% | +1873% | 1.464 |

## 빠른 시작

```bash
pip install -r requirements.txt
python main.py
python main.py --symbol ETH/USDT:USDT --tf 1h --start 2024-01-01 --end 2025-01-01
```

## 도구

```bash
# 파라미터 최적화
python optimize.py --mode random -n 200 --target all

# 멀티 심볼 스캔
python scan.py --symbols BTC ETH SOL --tf 15m 1h

# 역추세 전략 비교 (6가지 변형)
python ct_compare.py

# 주중 필터 비교
python weekday_compare.py
```

## 설정

`config/params.json`을 편집하여 심볼, 타임프레임, 기간, 전략 파라미터를 변경합니다.

## 프로젝트 구조

```
├── main.py               # 메인 백테스트 실행
├── ct_compare.py         # 역추세 6종 비교
├── optimize.py           # 그리드/랜덤 파라미터 탐색
├── scan.py               # 멀티 심볼 스캔
├── weekday_compare.py    # 주중 필터 비교
├── config/
│   └── params.json       # 전략 파라미터
├── data/
│   └── fetcher.py        # OKX OHLCV 데이터 수집 (parquet 캐시)
├── indicators/
│   ├── supertrend.py     # Supertrend (TradingView 동일 로직)
│   ├── rsi_filter.py     # RSI 상태머신 필터
│   ├── grad_filter.py    # BB 기울기 필터
│   └── counter_signals.py # 역추세 신호
├── strategy/
│   └── signal.py         # 신호 생성
└── backtest/
    ├── engine.py          # 백테스트 엔진
    └── reporter.py        # 성과 리포트
```

## 요구사항

- Python 3.10+
- ccxt, pandas, numpy, tabulate, python-dotenv, pyarrow
