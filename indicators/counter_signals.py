"""
역추세(Counter-Trend) 신호 계산

역추세 롱 (Supertrend가 SHORT 방향): ST 하락추세 + N연속 음봉 + RSI 과매도
역추세 숏 (Supertrend가 LONG  방향): ST 상승추세 + N연속 양봉 + RSI 과매수

진입 철학:
  - 강한 추세와 반대로 진입하므로 RSI 극단 필수
  - DCA는 RSI가 더 극단으로 이동할 때만 허용 (추세 가속 방지)
  - 청산: ST 역전 복귀 OR RSI 회복 중 먼저 도달하는 것
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
        if close[i] < open_[i]:       # 음봉
            consec[i] = consec[i - 1] - 1 if consec[i - 1] < 0 else -1
        elif close[i] > open_[i]:     # 양봉
            consec[i] = consec[i - 1] + 1 if consec[i - 1] > 0 else 1
        # 도지: 0 유지

    return pd.Series(consec, index=df.index, name="ct_consec")


def add_counter_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    역추세 신호 컬럼 추가.
    st1_dir, st2_dir 컬럼이 이미 있어야 함 (build_signals 이후 호출).

    추가 컬럼:
      ct_rsi         : RSI 값 (DCA 판단용)
      ct_consec      : 연속 봉 수 (음봉 음수, 양봉 양수)
      ct_long_entry  : 역추세 롱 1차 진입 신호
      ct_short_entry : 역추세 숏 1차 진입 신호
      ct_close_long  : 역추세 롱 청산 신호 (ST 복귀 or RSI 회복)
      ct_close_short : 역추세 숏 청산 신호
    """
    df = df.copy()

    rsi    = compute_rsi(df["close"], cfg["rsi_period"])
    consec = compute_consecutive_candles(df)

    df["ct_rsi"]    = rsi
    df["ct_consec"] = consec

    n_candles      = cfg["consec_candles"]
    rsi_long_e1    = cfg["rsi_long_entry1"]
    rsi_short_e1   = cfg["rsi_short_entry1"]
    rsi_long_exit  = cfg["rsi_long_exit"]
    rsi_short_exit = cfg["rsi_short_exit"]

    st_short = (df["st1_dir"] > 0) & (df["st2_dir"] > 0)
    st_long  = (df["st1_dir"] < 0) & (df["st2_dir"] < 0)

    # 1차 진입: ST방향 확인 + 연속 음봉/양봉 + RSI 극단
    df["ct_long_entry"]  = st_short & (consec <= -n_candles) & (rsi <= rsi_long_e1)
    df["ct_short_entry"] = st_long  & (consec >= n_candles)  & (rsi >= rsi_short_e1)

    # 청산: ST 역전(복귀) OR RSI 회복
    df["ct_close_long"]  = st_long  | (rsi >= rsi_long_exit)
    df["ct_close_short"] = st_short | (rsi <= rsi_short_exit)

    return df
