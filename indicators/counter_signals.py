"""
역추세(Counter-Trend) 신호 계산

진입 전략 (파인스크립트 long역추세.pine 기반):
  첫 진입: ST 반대방향 + 연속 음봉 N개 + EMA 필터 + 음봉 크기 필터
  DCA   : RSI 불리시 다이버전스 + 직전 매수가 대비 N% 추가 하락

바닥 품질 점수 시스템 (ct_bottom_quality, 0~3):
  1. 아래꼬리 비율 ≥ 40%  → 봉 안에서 매수세 유입 확인
  2. 거래량 스파이크 ≥ 1.5×평균 → 매도 탈진(capitulation)
  3. 캔들 바디 수축 → 하락 모멘텀 둔화

RSI 불리시 다이버전스:
  가격 저점이 이전 저점보다 낮은데 RSI 저점은 이전보다 높음
  → 하락 모멘텀 약화, 반등 가능성 시그널
"""

import numpy as np
import pandas as pd

from indicators.rsi_filter import compute_rsi


def compute_consecutive_candles(df: pd.DataFrame) -> pd.Series:
    """
    연속 음봉/양봉 수 계산.
    음봉 연속: 음수 (음봉 4개 연속 → -4)
    양봉 연속: 양수 (양봉 4개 연속 → +4)
    도지(close == open): 카운터 리셋
    """
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)
    consec = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        if close[i] < open_[i]:
            consec[i] = consec[i - 1] - 1 if consec[i - 1] < 0 else -1
        elif close[i] > open_[i]:
            consec[i] = consec[i - 1] + 1 if consec[i - 1] > 0 else 1

    return pd.Series(consec, index=df.index, name="ct_consec")


def compute_bottom_quality(df: pd.DataFrame) -> pd.Series:
    """
    바닥 품질 점수 (0~3점) - 반전 가능성 종합 평가

    구성요소 (각 1점):
      1. 아래꼬리 비율 ≥ 40%  : 봉 내 매수세 유입
      2. 거래량 스파이크      : vol ≥ 1.5 × 20봉평균
      3. 캔들 바디 수축       : 이전 봉보다 body 작음
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    vol   = df["volume"].values
    n     = len(df)

    total_range = high - low
    body        = np.abs(close - open_)
    lower_wick  = np.minimum(open_, close) - low

    # 1. 아래꼬리 비율 ≥ 40%
    wick_ratio = np.where(total_range > 1e-12, lower_wick / total_range, 0.0)
    has_wick = wick_ratio >= 0.40

    # 2. 거래량 스파이크 (20봉 이동평균 × 1.5)
    avg_vol = pd.Series(vol).rolling(20, min_periods=5).mean().values
    has_vol_spike = vol >= avg_vol * 1.5

    # 3. 캔들 바디 수축 (전봉 대비)
    prev_body = np.empty(n)
    prev_body[0] = body[0]
    prev_body[1:] = body[:-1]
    has_body_shrink = body < prev_body

    score = (has_wick.astype(int)
             + has_vol_spike.astype(int)
             + has_body_shrink.astype(int))

    return pd.Series(score, index=df.index, name="ct_bottom_quality")


def compute_rsi_divergence(
    df: pd.DataFrame,
    rsi: pd.Series,
    pivot_period: int = 5,
    max_bars: int = 60,
) -> tuple[pd.Series, pd.Series]:
    """
    RSI 다이버전스 감지 (파인스크립트 LonesomeTheBlue 로직 기반)

    불리시 다이버전스 (롱 DCA 신호):
      - 가격 저점이 이전 피봇 저점보다 낮음
      - RSI 저점은 이전 피봇 RSI보다 높음
      → 하락 모멘텀 약화

    베어리시 다이버전스 (숏 DCA 신호):
      - 가격 고점이 이전 피봇 고점보다 높음
      - RSI 고점은 이전 피봇 RSI보다 낮음
      → 상승 모멘텀 약화
    """
    n      = len(df)
    low    = df["low"].values
    high   = df["high"].values
    rsi_v  = rsi.values

    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)

    # 피봇 저점/고점 수집
    pl_idx, pl_low, pl_rsi = [], [], []
    ph_idx, ph_high, ph_rsi = [], [], []

    for i in range(pivot_period, n - pivot_period):
        lo_slice = low[i - pivot_period: i + pivot_period + 1]
        if low[i] <= np.min(lo_slice):
            pl_idx.append(i)
            pl_low.append(low[i])
            pl_rsi.append(rsi_v[i])

        hi_slice = high[i - pivot_period: i + pivot_period + 1]
        if high[i] >= np.max(hi_slice):
            ph_idx.append(i)
            ph_high.append(high[i])
            ph_rsi.append(rsi_v[i])

    # 불리시 다이버전스: 현재 피봇저점 < 이전, RSI 저점 > 이전
    for k in range(1, len(pl_idx)):
        ci, pi = pl_idx[k], pl_idx[k - 1]
        bar_gap = ci - pi
        if bar_gap < 5 or bar_gap > max_bars:
            continue
        if pl_low[k] < pl_low[k - 1] and pl_rsi[k] > pl_rsi[k - 1]:
            sig = min(ci + pivot_period, n - 1)
            bull_div[sig] = True

    # 베어리시 다이버전스: 현재 피봇고점 > 이전, RSI 고점 < 이전
    for k in range(1, len(ph_idx)):
        ci, pi = ph_idx[k], ph_idx[k - 1]
        bar_gap = ci - pi
        if bar_gap < 5 or bar_gap > max_bars:
            continue
        if ph_high[k] > ph_high[k - 1] and ph_rsi[k] < ph_rsi[k - 1]:
            sig = min(ci + pivot_period, n - 1)
            bear_div[sig] = True

    return (
        pd.Series(bull_div, index=df.index, name="rsi_bull_div"),
        pd.Series(bear_div, index=df.index, name="rsi_bear_div"),
    )


def add_counter_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    역추세 신호 컬럼 추가.
    st1_dir, st2_dir 컬럼이 이미 있어야 함 (build_signals 이후 호출).

    추가 컬럼:
      ct_rsi            : RSI 값
      ct_consec         : 연속 봉 수 (음봉 음수, 양봉 양수)
      ct_bottom_quality : 바닥 품질 점수 0~3
      rsi_bull_div      : RSI 불리시 다이버전스 신호
      rsi_bear_div      : RSI 베어리시 다이버전스 신호
      ct_long_entry     : 역추세 롱 진입 신호
      ct_short_entry    : 역추세 숏 진입 신호
      ct_close_long     : 역추세 롱 청산 신호 (ST 복귀)
      ct_close_short    : 역추세 숏 청산 신호
    """
    df = df.copy()

    rsi    = compute_rsi(df["close"], cfg["rsi_period"])
    consec = compute_consecutive_candles(df)
    bq     = compute_bottom_quality(df)

    # RSI 다이버전스
    pivot_period = cfg.get("divergence_pivot_period", 5)
    max_bars     = cfg.get("divergence_max_bars", 60)
    bull_div, bear_div = compute_rsi_divergence(df, rsi, pivot_period, max_bars)

    df["ct_rsi"]            = rsi
    df["ct_consec"]         = consec
    df["ct_bottom_quality"] = bq
    df["rsi_bull_div"]      = bull_div
    df["rsi_bear_div"]      = bear_div

    # EMA 필터 (파인스크립트의 ema_check)
    ema_cfg    = cfg.get("ema_filter", {})
    ema_enable = ema_cfg.get("enabled", False)
    ema_len    = ema_cfg.get("length", 30)
    if ema_enable:
        ema_val = df["close"].ewm(span=ema_len, adjust=False).mean()
        ct_ema_long  = df["close"] <= ema_val   # 롱: EMA 이하에서 진입
        ct_ema_short = df["close"] >= ema_val   # 숏: EMA 이상에서 진입
    else:
        ct_ema_long  = pd.Series(True, index=df.index)
        ct_ema_short = pd.Series(True, index=df.index)

    # 음봉 최소 크기 필터 (파인스크립트의 largeRedCandle)
    min_candle_pct = cfg.get("min_candle_pct", 0.0)
    if min_candle_pct > 0:
        candle_size = (df["open"] - df["close"]).abs() / df["open"]
        large_candle = candle_size >= min_candle_pct
    else:
        large_candle = pd.Series(True, index=df.index)

    n_candles      = cfg["consec_candles"]
    n_candles_long  = cfg.get("consec_candles_long",  n_candles)
    n_candles_short = cfg.get("consec_candles_short", n_candles)
    rsi_long_e1    = cfg["rsi_long_entry1"]
    rsi_short_e1   = cfg["rsi_short_entry1"]
    rsi_long_exit  = cfg.get("rsi_long_exit",  65)
    rsi_short_exit = cfg.get("rsi_short_exit", 35)
    quality_min       = cfg.get("ct_bottom_quality_min", 2)
    quality_min_long  = cfg.get("ct_bottom_quality_min_long",  quality_min)
    quality_min_short = cfg.get("ct_bottom_quality_min_short", quality_min)

    st_short = (df["st1_dir"] > 0) & (df["st2_dir"] > 0)
    st_long  = (df["st1_dir"] < 0) & (df["st2_dir"] < 0)

    # ── 첫 진입 신호 ─────────────────────────────────────────────────────
    # 파인스크립트: redCandleCount >= 4 + ema_check + RSI 극단 (선택)
    ct_long_entry  = (st_short
                      & (consec <= -n_candles_long)
                      & (rsi <= rsi_long_e1)
                      & ct_ema_long
                      & large_candle)

    ct_short_entry = (st_long
                      & (consec >= n_candles_short)
                      & (rsi >= rsi_short_e1)
                      & ct_ema_short
                      & large_candle)

    # Quality 경로: 봉 1개 완화 + RSI 2pt 강화 + 바닥 품질 신호 필수
    ct_long_quality  = (st_short
                        & (consec <= -(n_candles_long - 1))
                        & (rsi <= rsi_long_e1 - 2)
                        & (bq >= quality_min_long)
                        & ct_ema_long)
    ct_short_quality = (st_long
                        & (consec >= (n_candles_short - 1))
                        & (rsi >= rsi_short_e1 + 2)
                        & (bq >= quality_min_short)
                        & ct_ema_short)

    # BTC 매크로 필터 (btc_bull/btc_bear 컬럼이 있으면 적용)
    if "btc_bull" in df.columns:
        ct_long_entry   = ct_long_entry   & df["btc_bull"]
        ct_long_quality = ct_long_quality & df["btc_bull"]
    if "btc_bear" in df.columns:
        ct_short_entry   = ct_short_entry   & df["btc_bear"]
        ct_short_quality = ct_short_quality & df["btc_bear"]

    ct_long_enabled  = cfg.get("ct_long_enabled",  True)
    ct_short_enabled = cfg.get("ct_short_enabled", True)

    df["ct_long_entry"]  = (ct_long_entry  | ct_long_quality)  & ct_long_enabled
    df["ct_short_entry"] = (ct_short_entry | ct_short_quality) & ct_short_enabled

    # ── 청산: ST 역전(복귀) / RSI 회복 ─────────────────────────────────
    # ct_exit_enabled=False이면 TP/SL만 사용 (파인스크립트 스타일)
    ct_exit_enabled = cfg.get("ct_exit_enabled", True)
    if ct_exit_enabled:
        df["ct_close_long"]  = st_long  | (rsi >= rsi_long_exit)
        df["ct_close_short"] = st_short | (rsi <= rsi_short_exit)
    else:
        df["ct_close_long"]  = pd.Series(False, index=df.index)
        df["ct_close_short"] = pd.Series(False, index=df.index)

    return df
