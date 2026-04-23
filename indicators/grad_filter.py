"""
볼린저밴드 기울기 필터 (횡보장 제거)

Pine 로직:
  g_basis     = ta.sma(close, 50)
  g_slope     = g_basis - g_basis[1]
  g_slope_pct = g_slope / g_basis[1] * 100
  g_is_active = math.abs(g_slope_pct) > 0.06   ← 활동성 있는 시장

  grad_filter_ok = grad_filter ? g_is_active : true
"""

import numpy as np
import pandas as pd


def compute_grad_filter(
    close: pd.Series,
    bb_length: int = 50,
    bb_mult: float = 2.0,
    threshold_pct: float = 0.06,
    enabled: bool = True,
) -> pd.Series:
    """
    볼린저밴드 기울기 필터

    Returns
    -------
    grad_ok : bool Series - True이면 진입 허용 (시장이 활동적)
    """
    if not enabled:
        return pd.Series(True, index=close.index, name="grad_filter_ok")

    basis     = close.rolling(bb_length).mean()
    slope     = basis - basis.shift(1)
    slope_pct = slope / basis.shift(1) * 100

    is_active = slope_pct.abs() > threshold_pct
    return is_active.fillna(False).rename("grad_filter_ok")
