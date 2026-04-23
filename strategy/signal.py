"""
신호 생성 모듈 - 모든 지표를 조합하여 진입/청산 시그널 생성

Pine 진입 조건 (active_bb_atr=false 기준):
  longentrycond1  = direction1 < 0 AND direction2 < 0 AND long_rsi_cond AND grad_filter_ok
  shortentrycond1 = direction1 > 0 AND direction2 > 0 AND short_rsi_cond AND grad_filter_ok

Pine 청산 조건:
  longclosecond1  = direction1 > 0 OR direction2 > 0  (슈퍼트렌드 역전)
  shortclosecond1 = direction1 < 0 OR direction2 < 0

SL 역전 조건 (청산 후 반대 방향 진입):
  long  SL 역전: position_size > 0 AND close < avg_price*(1-lsl) AND grad_filter_ok → SHORT 진입
  short SL 역전: position_size < 0 AND close > avg_price*(1+ssl) AND grad_filter_ok → LONG  진입
"""

import pandas as pd

from indicators.supertrend import add_supertrend_columns
from indicators.rsi_filter  import compute_rsi_filter_states
from indicators.grad_filter import compute_grad_filter


def build_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    DataFrame에 모든 지표 + 신호 컬럼을 추가하여 반환

    추가 컬럼:
      st1_dir, st2_dir         : 슈퍼트렌드 방향 (-1=상승, +1=하락)
      long_rsi_cond            : RSI 필터 롱 허용
      short_rsi_cond           : RSI 필터 숏 허용
      grad_filter_ok           : 기울기 필터 허용
      long_entry               : 롱 진입 신호
      short_entry              : 숏 진입 신호
      close_long               : 롱 청산 신호 (슈퍼트렌드 역전)
      close_short              : 숏 청산 신호 (슈퍼트렌드 역전)
    """
    df = df.copy()

    # ── Supertrend 1, 2 ──────────────────────────────────────────────
    st1 = params["st1"]
    st2 = params["st2"]

    df = add_supertrend_columns(df, st1["factor"], st1["atr_period"], prefix="st1")
    df = add_supertrend_columns(df, st2["factor"], st2["atr_period"], prefix="st2")

    # ── RSI 필터 ─────────────────────────────────────────────────────
    rsi_cfg = params["rsi_filter"]
    long_rsi, short_rsi = compute_rsi_filter_states(
        df["close"],
        rsi_length    = rsi_cfg["rsi_length"],
        rsi_ma_length = rsi_cfg["rsi_ma_length"],
        long_block    = rsi_cfg["long_block"],
        long_unblock  = rsi_cfg["long_unblock"],
        short_block   = rsi_cfg["short_block"],
        short_unblock = rsi_cfg["short_unblock"],
        enabled       = rsi_cfg["enabled"],
    )
    df["long_rsi_cond"]  = long_rsi
    df["short_rsi_cond"] = short_rsi

    # ── 기울기 필터 ──────────────────────────────────────────────────
    grad_cfg = params["grad_filter"]
    df["grad_filter_ok"] = compute_grad_filter(
        df["close"],
        bb_length     = grad_cfg["bb_length"],
        bb_mult       = grad_cfg["bb_mult"],
        threshold_pct = grad_cfg["threshold_pct"],
        enabled       = grad_cfg["enabled"],
    )

    # ── 진입 신호 ────────────────────────────────────────────────────
    # 롱: 두 ST 모두 상승추세(dir < 0) + RSI 필터 허용 + 기울기 허용
    df["long_entry"] = (
        (df["st1_dir"] < 0) &
        (df["st2_dir"] < 0) &
        df["long_rsi_cond"] &
        df["grad_filter_ok"]
    )

    # 숏: 두 ST 모두 하락추세(dir > 0) + RSI 필터 허용 + 기울기 허용
    df["short_entry"] = (
        (df["st1_dir"] > 0) &
        (df["st2_dir"] > 0) &
        df["short_rsi_cond"] &
        df["grad_filter_ok"]
    )

    # ── 청산 신호 (슈퍼트렌드 역전) ──────────────────────────────────
    # 롱 보유 중 ST1 또는 ST2 하락 전환 → 청산
    df["close_long"]  = (df["st1_dir"] > 0) | (df["st2_dir"] > 0)
    # 숏 보유 중 ST1 또는 ST2 상승 전환 → 청산
    df["close_short"] = (df["st1_dir"] < 0) | (df["st2_dir"] < 0)

    # ── 역추세 신호 ──────────────────────────────────────────────────
    ct_cfg = params.get("counter_trend", {})
    if ct_cfg.get("enabled", False):
        from indicators.counter_signals import add_counter_signals
        df = add_counter_signals(df, ct_cfg)

    return df
