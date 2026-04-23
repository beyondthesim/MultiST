"""
Supertrend 지표 계산 - TradingView ta.supertrend() 와 동일한 로직

핵심 차이점:
- ATR = RMA(TR, period)  ← Wilder's Moving Average (EWM alpha=1/period)
- 첫 period 개 값은 SMA로 시드, 이후 Wilder 방식 적용
- Final band 계산: 이전 close 기준으로 밴드 클리핑
- Direction: -1 = 상승추세 (long), +1 = 하락추세 (short)
"""

import numpy as np
import pandas as pd


def rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's Moving Average (RMA) - TradingView ta.rma() 와 동일
    첫 period 개 값의 SMA를 시드로 사용
    """
    arr = series.values.astype(float)
    result = np.full(len(arr), np.nan)

    # 첫 번째 유효 시드: period 번째까지의 SMA
    # NaN이 있을 수 있으므로 첫 연속 유효 구간 찾기
    first_valid = 0
    while first_valid < len(arr) and np.isnan(arr[first_valid]):
        first_valid += 1

    seed_end = first_valid + period
    if seed_end > len(arr):
        return pd.Series(result, index=series.index)

    result[seed_end - 1] = np.nanmean(arr[first_valid:seed_end])

    for i in range(seed_end, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = (result[i - 1] * (period - 1) + arr[i]) / period

    return pd.Series(result, index=series.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range - ta.tr(true) 와 동일 (전봉 종가 포함)"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    # 첫 번째 봉은 prev_close가 없으므로 high-low만 사용
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    factor: float,
    atr_period: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend 계산

    Returns
    -------
    st_line  : Supertrend 라인값
    direction: -1 = 상승추세(Long), +1 = 하락추세(Short)
    """
    tr  = true_range(high, low, close)
    atr = rma(tr, atr_period)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr

    n = len(close)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction   = np.full(n, np.nan)
    st_line     = np.full(n, np.nan)

    bu = basic_upper.values
    bl = basic_lower.values
    cl = close.values

    for i in range(n):
        if np.isnan(atr.values[i]):
            continue

        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = bu[i]
            final_lower[i] = bl[i]
        else:
            # Final Upper: 새 upper가 이전보다 낮거나, 이전 close가 이전 upper보다 위이면 갱신
            if bu[i] < final_upper[i - 1] or cl[i - 1] > final_upper[i - 1]:
                final_upper[i] = bu[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Final Lower: 새 lower가 이전보다 높거나, 이전 close가 이전 lower보다 아래이면 갱신
            if bl[i] > final_lower[i - 1] or cl[i - 1] < final_lower[i - 1]:
                final_lower[i] = bl[i]
            else:
                final_lower[i] = final_lower[i - 1]

        # Direction 결정
        if i == 0 or np.isnan(direction[i - 1]):
            direction[i] = 1  # 초기값: 하락추세
        else:
            prev_dir = direction[i - 1]
            prev_st  = st_line[i - 1]

            if prev_st == final_upper[i - 1]:
                # 이전에 하락추세였음
                direction[i] = -1 if cl[i] > final_upper[i] else 1
            else:
                # 이전에 상승추세였음
                direction[i] = 1 if cl[i] < final_lower[i] else -1

        st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]

    return (
        pd.Series(st_line,   index=close.index, name="supertrend"),
        pd.Series(direction, index=close.index, name="direction"),
    )


def add_supertrend_columns(
    df: pd.DataFrame,
    factor: float,
    atr_period: int,
    prefix: str = "st",
) -> pd.DataFrame:
    """DataFrame에 supertrend 컬럼 추가 (in-place 아닌 복사본 반환)"""
    st, direction = supertrend(df["high"], df["low"], df["close"], factor, atr_period)
    df = df.copy()
    df[f"{prefix}_line"] = st
    df[f"{prefix}_dir"]  = direction
    return df
