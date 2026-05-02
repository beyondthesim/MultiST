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

# pandas resample alias 매핑
_TF_ALIAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "12m": "12min",
    "15m": "15min", "30m": "30min", "45m": "45min",
    "1h": "1h", "2h": "2h", "4h": "4h",
    "60m": "1h", "120m": "2h", "240m": "4h",
}


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """df(DatetimeIndex OHLCV)를 tf 주기로 리샘플링"""
    alias = _TF_ALIAS.get(tf, tf)
    return (
        df[["open", "high", "low", "close", "volume"]]
        .resample(alias, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )


def _build_ct_signals_htf(
    df_base: pd.DataFrame,
    ct_cfg: dict,
    params: dict,
    ct_tf: str,
) -> pd.DataFrame:
    """
    상위 타임프레임(ct_tf)에서 CT 신호를 계산하고 df_base 인덱스로 매핑.

    - 진입 신호: no-ffill (HTF 봉 확정 시점 한 번만 발화)
    - 청산 신호: ffill  (다음 HTF 봉까지 유지)
    """
    from indicators.counter_signals import add_counter_signals

    df_ht = _resample_ohlcv(df_base, ct_tf)

    # HTF에서 독립적인 Supertrend 계산
    st1 = params["st1"]
    st2 = params["st2"]
    df_ht = add_supertrend_columns(df_ht, st1["factor"], st1["atr_period"], prefix="st1")
    df_ht = add_supertrend_columns(df_ht, st2["factor"], st2["atr_period"], prefix="st2")

    # HTF CT 신호 계산
    df_ht = add_counter_signals(df_ht, ct_cfg)

    # 진입 신호는 당봉 종가에 확정 → 다음 봉에 실행 (shift 1)
    for col in ("ct_long_entry", "ct_short_entry"):
        df_ht[col] = df_ht[col].shift(1).fillna(False)

    result = df_base.copy()

    # 진입 신호: no-ffill (HTF 봉 시작 순간만 True)
    entry_cols = ["ct_long_entry", "ct_short_entry", "rsi_bull_div", "rsi_bear_div"]
    for col in entry_cols:
        if col in df_ht.columns:
            result[col] = (
                df_ht[col]
                .reindex(result.index)
                .fillna(False)
                .astype(bool)
            )

    # 청산 신호: ffill (다음 HTF 봉 시작까지 유지)
    close_cols = ["ct_close_long", "ct_close_short"]
    for col in close_cols:
        if col in df_ht.columns:
            result[col] = (
                df_ht[col]
                .reindex(result.index, method="ffill")
                .fillna(False)
                .astype(bool)
            )

    # 보조 컬럼: ffill
    info_cols = ["ct_rsi", "ct_consec", "ct_bottom_quality"]
    for col in info_cols:
        if col in df_ht.columns:
            result[col] = (
                df_ht[col]
                .reindex(result.index, method="ffill")
                .fillna(0)
            )

    return result


def _add_btc_macro_cols(df_base: pd.DataFrame, btc_df: pd.DataFrame, btc_cfg: dict) -> pd.DataFrame:
    """BTC 매크로 방향성 컬럼 추가 (btc_bull, btc_bear) — ffill로 12m에 매핑"""
    ema_len      = btc_cfg.get("ema_length", 50)
    ema          = btc_df["close"].ewm(span=ema_len, adjust=False).mean()
    btc_bull_raw = btc_df["close"] > ema
    btc_bear_raw = btc_df["close"] < ema

    df = df_base.copy()
    df["btc_bull"] = (
        btc_bull_raw.reindex(df.index, method="ffill").fillna(False).astype(bool)
    )
    df["btc_bear"] = (
        btc_bear_raw.reindex(df.index, method="ffill").fillna(False).astype(bool)
    )
    return df


def build_signals_split_tf(
    df_base: pd.DataFrame,
    params: dict,
    main_tf: str,
    btc_df=None,
) -> pd.DataFrame:
    """
    Split TF 모드: base = CT TF (낮음, 예: 1m), 메인 신호는 main_tf(예: 12m)에서 계산 후 매핑.

    동작:
      - 메인 ST/RSI/Grad/entry/close: main_tf로 리샘플 후 계산 → base에 매핑
        * entry: 메인 봉 확정 후 다음 메인 봉 시작 시점에만 True (no-ffill)
        * close, st_dir, 필터: 다음 메인 봉까지 ffill (그 동안 SL/TP 평가됨)
      - CT 신호: base TF에서 직접 계산 (매핑된 메인 ST 방향 사용 → 메인 추세 반대로 진입)
      - BTC 매크로: 그대로 ffill

    base TF에서 engine이 매 봉마다 SL/TP/CT 평가 → 더 빠른 청산 가능.
    """
    df = df_base.copy()

    # ── 1. 메인 TF로 리샘플 ───────────────────────────────────────
    df_main = _resample_ohlcv(df_base, main_tf)

    # ── 2. 메인 ST/RSI/Grad on main_tf ────────────────────────────
    st1 = params["st1"]
    st2 = params["st2"]
    df_main = add_supertrend_columns(df_main, st1["factor"], st1["atr_period"], prefix="st1")
    df_main = add_supertrend_columns(df_main, st2["factor"], st2["atr_period"], prefix="st2")

    rsi_cfg = params["rsi_filter"]
    long_rsi_main, short_rsi_main = compute_rsi_filter_states(
        df_main["close"],
        rsi_length    = rsi_cfg["rsi_length"],
        rsi_ma_length = rsi_cfg["rsi_ma_length"],
        long_block    = rsi_cfg["long_block"],
        long_unblock  = rsi_cfg["long_unblock"],
        short_block   = rsi_cfg["short_block"],
        short_unblock = rsi_cfg["short_unblock"],
        enabled       = rsi_cfg["enabled"],
    )
    df_main["long_rsi_cond"]  = long_rsi_main
    df_main["short_rsi_cond"] = short_rsi_main

    grad_cfg = params["grad_filter"]
    df_main["grad_filter_ok"] = compute_grad_filter(
        df_main["close"],
        bb_length     = grad_cfg["bb_length"],
        bb_mult       = grad_cfg["bb_mult"],
        threshold_pct = grad_cfg["threshold_pct"],
        enabled       = grad_cfg["enabled"],
    )

    # 메인 entry/close 신호 (main_tf에서 계산)
    df_main["long_entry"] = (
        (df_main["st1_dir"] < 0) & (df_main["st2_dir"] < 0)
        & df_main["long_rsi_cond"] & df_main["grad_filter_ok"]
    )
    df_main["short_entry"] = (
        (df_main["st1_dir"] > 0) & (df_main["st2_dir"] > 0)
        & df_main["short_rsi_cond"] & df_main["grad_filter_ok"]
    )
    df_main["close_long"]  = (df_main["st1_dir"] > 0) | (df_main["st2_dir"] > 0)
    df_main["close_short"] = (df_main["st1_dir"] < 0) | (df_main["st2_dir"] < 0)

    # ── 3. base에 매핑 ───────────────────────────────────────────
    # entry: shift(1) → 봉 확정 후 다음 봉에 실행. no-ffill (메인 봉 시작 1m 봉에만 True)
    for col in ("long_entry", "short_entry"):
        df_main[col] = df_main[col].shift(1).fillna(False)
        df[col] = (
            df_main[col]
            .reindex(df.index)
            .fillna(False)
            .astype(bool)
        )

    # close, st_dir, 필터: ffill (메인 봉 단위로 유지)
    for col in ("close_long", "close_short", "st1_dir", "st2_dir",
                "long_rsi_cond", "short_rsi_cond", "grad_filter_ok"):
        if col.startswith("st") and col.endswith("dir"):
            df[col] = df_main[col].reindex(df.index, method="ffill").fillna(0)
        else:
            df[col] = df_main[col].reindex(df.index, method="ffill").fillna(False).astype(bool)

    # 메인 봉 close 시점 마킹 (engine 옵션: 메인 SL/TP만 이 시점에 평가)
    df["is_main_close_bar"] = df.index.isin(df_main.index)

    # ── 4. BTC 매크로 ────────────────────────────────────────────
    btc_cfg = params.get("btc_filter", {})
    if btc_cfg.get("enabled", False) and btc_df is not None:
        df = _add_btc_macro_cols(df, btc_df, btc_cfg)

    # ── 5. CT 신호 (base = 1m에서 직접 계산) ─────────────────────
    # 매핑된 메인 ST(st1_dir, st2_dir) 방향을 그대로 사용 → 메인 추세 반대로 CT 진입
    ct_cfg = params.get("counter_trend", {})
    if ct_cfg.get("enabled", False):
        from indicators.counter_signals import add_counter_signals
        df = add_counter_signals(df, ct_cfg)

    return df


def build_signals(df: pd.DataFrame, params: dict, btc_df=None) -> pd.DataFrame:
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

    # ── BTC 매크로 필터 ──────────────────────────────────────────────
    btc_cfg = params.get("btc_filter", {})
    if btc_cfg.get("enabled", False) and btc_df is not None:
        df = _add_btc_macro_cols(df, btc_df, btc_cfg)

    # ── 역추세 신호 ──────────────────────────────────────────────────
    ct_cfg = params.get("counter_trend", {})
    if ct_cfg.get("enabled", False):
        ct_tf   = ct_cfg.get("timeframe", "")
        main_tf = params.get("timeframe", "")

        if ct_tf and ct_tf != main_tf:
            # 상위 타임프레임 CT 신호
            df = _build_ct_signals_htf(df, ct_cfg, params, ct_tf)
        else:
            from indicators.counter_signals import add_counter_signals
            df = add_counter_signals(df, ct_cfg)

    return df
