"""
신호 계산 - 백테스트의 build_signals 그대로 재사용

새 봉 close 시점에 호출:
  - 메인 신호: long_entry, short_entry, close_long, close_short
  - CT 신호:   ct_long_entry, ct_short_entry, ct_close_long, ct_close_short
  - BTC 매크로: btc_bull, btc_bear (필터)

반환: 마지막 봉의 신호 딕셔너리
"""
import pandas as pd

from strategy.signal import build_signals
from live_bot import config, notifier
from live_bot.data_stream import fetch_resampled
from live_bot.exchange import OKXClient


def compute_latest_signals(client: OKXClient, params: dict) -> dict:
    """
    최신 봉까지 신호 계산 후 마지막 봉의 신호 반환.

    return: {
      "ts":               pd.Timestamp,
      "close":            float,
      "long_entry":       bool,
      "short_entry":      bool,
      "close_long":       bool,
      "close_short":      bool,
      "ct_long_entry":    bool,
      "ct_short_entry":   bool,
      "ct_close_long":    bool,
      "ct_close_short":   bool,
      "btc_bull":         bool (없으면 None),
      "btc_bear":         bool (없으면 None),
    }
    """
    symbol = params["symbol"]
    tf     = params["timeframe"]

    # 메인 OHLCV
    df = fetch_resampled(client, symbol, tf, config.WARMUP_BARS)
    if df.empty:
        notifier.error("OHLCV 데이터 없음")
        return {}

    # BTC 4h (BTC 필터 활성화 시)
    btc_cfg = params.get("btc_filter", {})
    df_btc = None
    if btc_cfg.get("enabled", False):
        btc_sym = btc_cfg.get("symbol", "BTC/USDT:USDT")
        btc_tf  = btc_cfg.get("timeframe", "4h")
        df_btc  = fetch_resampled(client, btc_sym, btc_tf, 250)

    # 신호 계산
    df_sig = build_signals(df, params, btc_df=df_btc)
    last = df_sig.iloc[-1]

    return {
        "ts":              df_sig.index[-1],
        "close":           float(last["close"]),
        "long_entry":      bool(last.get("long_entry",  False)),
        "short_entry":     bool(last.get("short_entry", False)),
        "close_long":      bool(last.get("close_long",  False)),
        "close_short":     bool(last.get("close_short", False)),
        "ct_long_entry":   bool(last.get("ct_long_entry",  False)),
        "ct_short_entry":  bool(last.get("ct_short_entry", False)),
        "ct_close_long":   bool(last.get("ct_close_long",  False)),
        "ct_close_short":  bool(last.get("ct_close_short", False)),
        "btc_bull":        bool(last["btc_bull"]) if "btc_bull" in df_sig.columns else None,
        "btc_bear":        bool(last["btc_bear"]) if "btc_bear" in df_sig.columns else None,
    }
