"""
A1~A5 + Baseline 옵션 평가
5개 대표 코인 평균 수익률/MDD 비교

A1: CT 사이즈만 1200→300 (1m base)
A2: CT 파라미터를 1m 시간 단위로 12배 튜닝 (1m base)
A3: 메인 SL/TP 메인TF 봉에서만 평가 (1m base, engine옵션)
A4: A1 + A2
A5: A1 + A3
Baseline: 12m base (현재)
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("C:/trade/DynamicDCAHedge/data/raw")
BTC_FILE = RAW_DIR / "BTC_USDT_USDT_4h.parquet"
COINS = ["ORDI", "MON", "DYDX", "BIO", "BASED"]
START_OFFSET = 7
END_OFFSET = 60


def _resample(df, tf):
    alias = {"1m": "1min", "12m": "12min"}[tf]
    return (df[["open","high","low","close","volume"]]
            .resample(alias, label="left", closed="left")
            .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
            .dropna(subset=["close"]))


def _stats(eq_curve, init_cap):
    if not eq_curve:
        return {"profit_pct": 0.0, "mdd": 0.0}
    final = eq_curve[-1]["equity"]
    profit_pct = (final - init_cap) / init_cap * 100
    peak = init_cap
    mdd = 0.0
    for pt in eq_curve:
        eq = pt["equity"]
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd
    return {"profit_pct": round(profit_pct, 2), "mdd": round(mdd, 1)}


def run_one(symbol, df_1m, btc_df, params, mode):
    """
    mode: 'baseline' (12m base) | 'split' (1m base + 12m main)
    """
    from strategy.signal import build_signals, build_signals_split_tf
    from backtest.engine import BacktestEngine

    listing = df_1m.index[0]
    bt_start = listing + pd.Timedelta(days=START_OFFSET)
    bt_end = min(df_1m.index[-1], listing + pd.Timedelta(days=END_OFFSET))
    if bt_end - bt_start < pd.Timedelta(days=14):
        return None

    p = dict(params)
    p["symbol"] = f"{symbol}/USDT:USDT"

    if mode == "baseline":
        df = _resample(df_1m, "12m")
        df = df[df.index < bt_end].copy()
        df["in_backtest"] = df.index >= bt_start
        if df["in_backtest"].sum() < 50:
            return None
        p["timeframe"] = "12m"
        try:
            df_sig = build_signals(df, p, btc_df=btc_df)
        except Exception as e:
            return None
    else:
        # split mode: base 1m, main 12m
        df = df_1m[df_1m.index < bt_end].copy()
        df = df[["open","high","low","close","volume"]].copy()
        df["in_backtest"] = df.index >= bt_start
        if df["in_backtest"].sum() < 100:
            return None
        p["timeframe"] = "12m"  # 메인 TF
        try:
            df_sig = build_signals_split_tf(df, p, main_tf="12m", btc_df=btc_df)
        except Exception as e:
            return None

    df_bt = df_sig[df_sig["in_backtest"]].copy()
    if df_bt.empty:
        return None

    try:
        engine = BacktestEngine(p)
        mt, ct, eq = engine.run(df_bt)
    except Exception as e:
        return None

    s = _stats(eq, p["initial_capital"])
    s["main_trades"] = len(mt)
    s["ct_trades"] = len(ct)
    return s


def make_params(base, opt):
    p = copy.deepcopy(base)
    ct = p["counter_trend"]
    if opt == "baseline":
        # 변경 없음 (12m baseline)
        pass
    elif opt == "A1":
        ct["ct_position_pct"] = 300
    elif opt == "A2":
        ct["rsi_period"] = 168
        ct["consec_candles_long"] = 60
        ct["consec_candles_short"] = 48
        ct["consec_candles"] = 48
        ct["ema_filter"]["length"] = 360
        ct["divergence_pivot_period"] = 60
        ct["divergence_max_bars"] = 720
    elif opt == "A3":
        p["main_eval_only_at_main_close"] = True  # marker (engine은 is_main_close_bar 컬럼으로 동작)
    elif opt == "A4":
        # A1 + A2
        ct["ct_position_pct"] = 300
        ct["rsi_period"] = 168
        ct["consec_candles_long"] = 60
        ct["consec_candles_short"] = 48
        ct["consec_candles"] = 48
        ct["ema_filter"]["length"] = 360
        ct["divergence_pivot_period"] = 60
        ct["divergence_max_bars"] = 720
    elif opt == "A5":
        # A1 + A3
        ct["ct_position_pct"] = 300
    return p


def main():
    base = json.load(open("config/params.json"))
    btc_df = None
    if BTC_FILE.exists():
        btc_df = pd.read_parquet(BTC_FILE)
        btc_df.index = pd.to_datetime(btc_df.index, utc=True)

    coin_data = {}
    for c in COINS:
        path = RAW_DIR / f"{c}_USDT_USDT_1m.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            coin_data[c] = df

    options = [
        ("Baseline (12m main+12m CT)", "baseline", "baseline"),
        ("A1 (1m CT, size↓ 300)", "A1", "split"),
        ("A2 (1m CT, params x12)", "A2", "split"),
        ("A3 (1m CT, main SL@12m only)", "A3", "split"),
        ("A4 (A1+A2)", "A4", "split"),
        ("A5 (A1+A3)", "A5", "split"),
    ]

    print(f"\n{'='*100}")
    print(f"  옵션 비교: {len(coin_data)}개 코인 평균 (대표: {COINS})")
    print(f"{'='*100}\n")

    summary = []
    for label, opt, mode in options:
        params = make_params(base, opt)
        coin_results = []
        t0 = time.time()
        for sym, df_1m in coin_data.items():
            r = run_one(sym, df_1m, btc_df, params, mode)
            if r:
                coin_results.append({"symbol": sym, **r})
        elapsed = time.time() - t0
        if not coin_results:
            print(f"{label:40s}  결과 없음")
            continue
        df_r = pd.DataFrame(coin_results)
        avg_p = df_r["profit_pct"].mean()
        med_p = df_r["profit_pct"].median()
        avg_m = df_r["mdd"].mean()
        max_m = df_r["mdd"].max()
        avg_mt = df_r["main_trades"].mean()
        avg_ct = df_r["ct_trades"].mean()
        win = int((df_r["profit_pct"] > 0).sum())
        score = avg_p / max(avg_m, 1.0)
        summary.append({
            "label": label,
            "opt": opt,
            "avg_profit": round(avg_p, 2),
            "med_profit": round(med_p, 2),
            "avg_mdd": round(avg_m, 1),
            "max_mdd": round(max_m, 1),
            "avg_main_t": round(avg_mt, 0),
            "avg_ct_t": round(avg_ct, 0),
            "win_count": f"{win}/{len(df_r)}",
            "score": round(score, 2),
            "elapsed_s": round(elapsed, 1),
        })
        # per coin 출력
        print(f"\n[{label}]  ({elapsed:.1f}s)")
        print(df_r.to_string(index=False))
        print(f"  avg profit: {avg_p:+7.2f}%  avg mdd: {avg_m:5.1f}%  win: {win}/{len(df_r)}  score: {score:+.2f}")

    # 종합
    print(f"\n{'='*100}")
    print(f"  종합 비교 (5개 코인 평균)")
    print(f"{'='*100}")
    if summary:
        df_sum = pd.DataFrame(summary)
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
