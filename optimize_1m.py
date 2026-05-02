"""
1m TF용 ST 파라미터 그리드 서치 — 신규상장 코인 다수에 대해 평균 성과 평가

사용법:
    python optimize_1m.py
    python optimize_1m.py --coins ORDI MON DYDX --top 10
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd

RAW_DIR = Path("C:/trade/DynamicDCAHedge/data/raw")
BTC_FILE = RAW_DIR / "BTC_USDT_USDT_4h.parquet"

# 평가 대상 코인 (수익률 다양한 분포로 대표성 확보)
DEFAULT_COINS = ["ORDI", "MON", "DYDX", "BIO", "BASED",
                 "RESOLV", "SAHARA", "TRIA", "WET", "OPN"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--coins", nargs="+", default=DEFAULT_COINS,
                    help="평가할 코인 목록 (기본: 10개)")
    ap.add_argument("--start-offset", type=int, default=7)
    ap.add_argument("--end-offset", type=int, default=60)
    ap.add_argument("--factors", nargs="+", type=float,
                    default=[8, 12, 16, 20, 25, 30],
                    help="ST factor 그리드")
    ap.add_argument("--atrs", nargs="+", type=int,
                    default=[60, 120, 180, 240, 360, 480],
                    help="ATR period 그리드 (1m 단위, 60=1h)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--sort", default="score",
                    choices=["score", "profit", "mdd", "pf"])
    return ap.parse_args()


def _stats(trades: list[dict], eq_curve: list[dict], init_cap: float) -> dict:
    if not trades:
        return {"trades": 0, "profit_pct": 0.0, "pf": 0.0, "mdd": 0.0}
    n = len(trades)
    gp = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gl = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    pf = gp / gl if gl > 0 else 999.0
    final = eq_curve[-1]["equity"] if eq_curve else init_cap
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
    return {"trades": n, "profit_pct": round(profit_pct, 2),
            "pf": round(pf, 3), "mdd": round(mdd, 1)}


def run_one(symbol: str, df_1m: pd.DataFrame, btc_df, params: dict,
            start_offset: int, end_offset: int) -> dict | None:
    from strategy.signal import build_signals
    from backtest.engine import BacktestEngine

    listing = df_1m.index[0]
    bt_start = listing + pd.Timedelta(days=start_offset)
    bt_end = min(df_1m.index[-1], listing + pd.Timedelta(days=end_offset))
    if bt_end - bt_start < pd.Timedelta(days=14):
        return None

    df = df_1m[df_1m.index < bt_end].copy()
    df["in_backtest"] = df.index >= bt_start
    if df["in_backtest"].sum() < 100:
        return None

    p = dict(params)
    p["symbol"] = f"{symbol}/USDT:USDT"
    p["timeframe"] = "1m"

    try:
        df_sig = build_signals(df, p, btc_df=btc_df)
        df_bt = df_sig[df_sig["in_backtest"]].copy()
        engine = BacktestEngine(p)
        mt, ct, eq = engine.run(df_bt)
    except Exception as e:
        return None

    init_cap = p["initial_capital"]
    main_s = _stats(mt, eq, init_cap)
    ct_s = _stats(ct, eq, init_cap)
    return {
        "main_trades": main_s["trades"],
        "ct_trades": ct_s["trades"],
        "main_profit": main_s["profit_pct"],
        "main_mdd": main_s["mdd"],
        "main_pf": main_s["pf"],
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        base_params = json.load(f)

    btc_df = None
    if base_params.get("btc_filter", {}).get("enabled", False) and BTC_FILE.exists():
        btc_df = pd.read_parquet(BTC_FILE)
        btc_df.index = pd.to_datetime(btc_df.index, utc=True)

    # 코인 데이터 미리 로드
    print(f"코인 로드: {args.coins}")
    coin_data = {}
    for sym in args.coins:
        path = RAW_DIR / f"{sym}_USDT_USDT_1m.parquet"
        if not path.exists():
            print(f"  {sym}: 파일 없음 — 스킵")
            continue
        df_1m = pd.read_parquet(path)
        df_1m.index = pd.to_datetime(df_1m.index, utc=True)
        coin_data[sym] = df_1m
    print(f"  로드 완료: {len(coin_data)}개\n")

    grid = list(product(args.factors, args.atrs))
    print(f"그리드: factor {args.factors} × atr {args.atrs} = {len(grid)} 조합")
    print(f"총 백테스트: {len(grid)} × {len(coin_data)} = {len(grid)*len(coin_data)}회\n")

    results = []
    t_start = time.time()

    for i, (factor, atr_p) in enumerate(grid, 1):
        params = copy.deepcopy(base_params)
        params["st1"] = {"atr_period": atr_p, "factor": factor}
        params["st2"] = {"atr_period": atr_p, "factor": factor}

        per_coin = []
        for sym, df_1m in coin_data.items():
            r = run_one(sym, df_1m, btc_df, params,
                        args.start_offset, args.end_offset)
            if r:
                per_coin.append(r)

        if not per_coin:
            continue

        df_per = pd.DataFrame(per_coin)
        avg_profit = df_per["main_profit"].mean()
        med_profit = df_per["main_profit"].median()
        avg_mdd = df_per["main_mdd"].mean()
        max_mdd = df_per["main_mdd"].max()
        avg_pf = df_per["main_pf"].replace(999.0, 5.0).mean()  # 999는 5로 캡
        avg_trades = df_per["main_trades"].mean()
        win_count = int((df_per["main_profit"] > 0).sum())

        # Score: profit_pct / mdd (Calmar-like)
        score = avg_profit / max(avg_mdd, 1.0)

        elapsed = time.time() - t_start
        eta = elapsed / i * (len(grid) - i)
        print(f"[{i:2d}/{len(grid)}] factor={factor:5.1f}  atr={atr_p:3d}  "
              f"profit avg {avg_profit:+7.2f}%  mdd {avg_mdd:5.1f}%  "
              f"pf {avg_pf:.2f}  trades {avg_trades:5.0f}  win {win_count}/{len(df_per)}  "
              f"score {score:+5.2f}  (ETA {eta:.0f}s)", flush=True)

        results.append({
            "factor": factor,
            "atr_period": atr_p,
            "avg_profit": round(avg_profit, 2),
            "med_profit": round(med_profit, 2),
            "avg_mdd": round(avg_mdd, 1),
            "max_mdd": round(max_mdd, 1),
            "avg_pf": round(avg_pf, 2),
            "avg_trades": round(avg_trades, 0),
            "win_count": win_count,
            "n_coins": len(df_per),
            "score": round(score, 2),
        })

    if not results:
        print("결과 없음")
        return

    df = pd.DataFrame(results)
    sort_col = {"score": "score", "profit": "avg_profit",
                "mdd": "avg_mdd", "pf": "avg_pf"}[args.sort]
    asc = args.sort == "mdd"
    df = df.sort_values(sort_col, ascending=asc)

    print(f"\n{'='*100}")
    print(f"  TOP {args.top} (정렬: {args.sort})")
    print(f"{'='*100}")
    print(df.head(args.top).to_string(index=False))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "optimize_1m_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
