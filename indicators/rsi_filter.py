"""
RSI 필터 - TradingView 파인스크립트와 동일한 상태머신 구현

Pine 로직:
  rsi_v  = ta.rsi(close, 14)          # RSI(14)
  rsi_ma = ta.sma(rsi_v, 2)           # RSI의 2봉 SMA

  var bool long_rsi_cond  = true
  var bool short_rsi_cond = true

  // 롱 필터
  if rsi_ma >= 75: long_rsi_cond  = false
  if long_rsi_cond  == false and rsi_ma <= 60: long_rsi_cond  = true

  // 숏 필터
  if rsi_ma <= 25: short_rsi_cond = false
  if short_rsi_cond == false and rsi_ma >= 40: short_rsi_cond = true
"""

import numpy as np
import pandas as pd

from indicators.supertrend import rma  # RMA 재사용


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 계산 - TradingView ta.rsi() 와 동일
    gain/loss 에 RMA(Wilder's) 적용
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0,   0)
    return rsi


def compute_rsi_filter_states(
    close: pd.Series,
    rsi_length: int  = 14,
    rsi_ma_length: int = 2,
    long_block: float   = 75,
    long_unblock: float = 60,
    short_block: float  = 25,
    short_unblock: float = 40,
    enabled: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    RSI 필터 상태머신 계산

    Returns
    -------
    long_cond  : bool Series - True이면 롱 진입 허용
    short_cond : bool Series - True이면 숏 진입 허용
    """
    n = len(close)

    if not enabled:
        ones = pd.Series(True, index=close.index)
        return ones, ones.copy()

    rsi_v  = compute_rsi(close, rsi_length)
    rsi_ma = rsi_v.rolling(rsi_ma_length).mean()

    long_arr  = np.ones(n, dtype=bool)
    short_arr = np.ones(n, dtype=bool)

    long_state  = True
    short_state = True

    rsi_ma_vals = rsi_ma.values

    for i in range(n):
        v = rsi_ma_vals[i]

        if not np.isnan(v):
            # 롱 필터 갱신
            if v >= long_block:
                long_state = False
            elif not long_state and v <= long_unblock:
                long_state = True

            # 숏 필터 갱신
            if v <= short_block:
                short_state = False
            elif not short_state and v >= short_unblock:
                short_state = True

        long_arr[i]  = long_state
        short_arr[i] = short_state

    return (
        pd.Series(long_arr,  index=close.index, name="long_rsi_cond"),
        pd.Series(short_arr, index=close.index, name="short_rsi_cond"),
    )
