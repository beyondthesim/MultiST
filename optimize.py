"""
파라미터 최적화 - 그리드 서치 / 랜덤 서치

최적화 가능 파라미터:
  Supertrend  : factor, atr
  RSI 필터    : long_block, long_unblock, short_block, short_unblock
  Grad 필터   : grad_threshold
  SL          : sl_pct (롱/숏 동일)

사용법:
  python optimize.py                          # 기본 (factor+ATR 그리드)
  python optimize.py --mode random -n 200     # 랜덤 서치 200회
  python optimize.py --target all             # 모든 파라미터 랜덤 서치
  python optimize.py --target st rsi          # Supertrend + RSI만
  python optimize.py --symbol ETH/USDT:USDT --tf 1h --target all -n 300
  python optimize.py --sort pf --top 20
"""

import argparse
import json
import random
import copy
from itertools import product
from pathlib import Path

import pandas as pd


# ── 파라미터 서치 공간 정의 ────────────────────────────────────────────────
SEARCH_SPACE = {
    "st": {
        "factor":     [3.0, 3.5, 4.0, 4.5, 5.0, 5.4, 6.0, 7.0, 8.0],
        "atr":        [8, 10, 12, 14, 16, 20],
    },
    "rsi": {
        "long_block":    [65, 70, 75, 80, 85],
        "long_unblock":  [45, 50, 55, 60, 65],
        "short_block":   [15, 20, 25, 30, 35],
        "short_unblock": [35, 40, 45, 50, 55],
    },
    "grad": {
        "threshold_pct": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    },
    "sl": {
        "sl_pct": [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Supertrend 파라미터 최적화")
    p.add_argument("--config",   default="config/params.json")
    p.add_argument("--symbol",   help="심볼 (예: ETH/USDT:USDT)")
    p.add_argument("--tf",       dest="timeframe", help="타임프레임")
    p.add_argument("--start",    dest="start_date")
    p.add_argument("--end",      dest="end_date")
    p.add_argument("--mode",     default="grid", choices=["grid", "random"],
                   help="서치 방법: grid(완전탐색) / random(랜덤)")
    p.add_argument("-n",         dest="n_trials", type=int, default=100,
                   help="랜덤 서치 시도 횟수 (--mode random 시)")
    p.add_argument("--target",   nargs="+",
                   default=["st"],
                   choices=["st", "rsi", "grad", "sl", "all"],
                   help="최적화 대상 파라미터 그룹 (기본: st)")
    p.add_argument("--sort",     default="profit",
                   choices=["profit", "pf", "winrate", "trades", "calmar"])
    p.add_argument("--top",      type=int, default=20)
    p.add_argument("--refresh",  action="store_true")
    p.add_argument("--min-trades", type=int, default=30)
    return p.parse_args()


def apply_params(base: dict, combo: dict) -> dict:
    """combo dict의 값들을 base params에 적용"""
    p = copy.deepcopy(base)

    if "factor" in combo:
        p["st1"]["factor"] = combo["factor"]
        p["st2"]["factor"] = combo["factor"]
    if "atr" in combo:
        p["st1"]["atr_period"] = combo["atr"]
        p["st2"]["atr_period"] = combo["atr"]

    if "long_block" in combo:
        p["rsi_filter"]["long_block"]    = combo["long_block"]
    if "long_unblock" in combo:
        p["rsi_filter"]["long_unblock"]  = combo["long_unblock"]
    if "short_block" in combo:
        p["rsi_filter"]["short_block"]   = combo["short_block"]
    if "short_unblock" in combo:
        p["rsi_filter"]["short_unblock"] = combo["short_unblock"]

    if "threshold_pct" in combo:
        p["grad_filter"]["threshold_pct"] = combo["threshold_pct"]

    if "sl_pct" in combo:
        p["long_sl_pct"]  = combo["sl_pct"]
        p["short_sl_pct"] = combo["sl_pct"]

    return p


def make_grid(targets: list[str]) -> list[dict]:
    """그리드 탐색: 선택된 그룹의 모든 조합 생성"""
    if "all" in targets:
        targets = ["st", "rsi", "grad", "sl"]

    keys_list = []
    vals_list = []
    for group in targets:
        space = SEARCH_SPACE[group]
        for k, v in space.items():
            keys_list.append(k)
            vals_list.append(v)

    combos = []
    for values in product(*vals_list):
        combo = dict(zip(keys_list, values))
        # RSI 유효성 체크: block > unblock
        if "long_block" in combo and "long_unblock" in combo:
            if combo["long_block"] <= combo["long_unblock"]:
                continue
        if "short_block" in combo and "short_unblock" in combo:
            if combo["short_block"] >= combo["short_unblock"]:
                continue
        combos.append(combo)
    return combos


def make_random(targets: list[str], n: int) -> list[dict]:
    """랜덤 서치: n개 조합 무작위 샘플링"""
    if "all" in targets:
        targets = ["st", "rsi", "grad", "sl"]

    combos = set()
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 20:
        attempts += 1
        combo = {}
        for group in targets:
            for k, v in SEARCH_SPACE[group].items():
                combo[k] = random.choice(v)

        # RSI 유효성
        if "long_block" in combo and "long_unblock" in combo:
            if combo["long_block"] <= combo["long_unblock"]:
                continue
        if "short_block" in combo and "short_unblock" in combo:
            if combo["short_block"] >= combo["short_unblock"]:
                continue

        key = tuple(sorted(combo.items()))
        if key not in combos:
            combos.add(key)
            results.append(combo)
    return results


def run_combo(df_raw, params_base: dict, combo: dict, min_trades: int) -> dict | None:
    from strategy.signal import build_signals
    from backtest.engine import BacktestEngine

    p = apply_params(params_base, combo)
    try:
        df_s  = build_signals(df_raw.copy(), p)
        df_bt = df_s[df_s["in_backtest"]].copy()

        engine = BacktestEngine(p)
        trades, _ct, eq_curve = engine.run(df_bt)

        if not trades or len(trades) < min_trades:
            return None

        n        = len(trades)
        winners  = sum(1 for t in trades if t["net_pnl"] > 0)
        init_cap = p["initial_capital"]
        final_eq = eq_curve[-1]["equity"] if eq_curve else init_cap
        profit_pct = (final_eq - init_cap) / init_cap * 100

        gp = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        gl = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
        pf = gp / gl if gl > 0 else 0.0

        peak = init_cap
        mdd  = 0.0
        for pt in eq_curve:
            eq   = pt["equity"]
            peak = max(peak, eq)
            dd   = (peak - eq) / peak * 100
            mdd  = max(mdd, dd)

        calmar = profit_pct / mdd if mdd > 0 else 0.0

        return {
            **combo,
            "trades":   n,
            "win_rate": round(winners / n * 100, 1),
            "profit":   round(profit_pct, 1),
            "pf":       round(pf, 3),
            "mdd":      round(mdd, 1),
            "calmar":   round(calmar, 2),
        }
    except Exception:
        return None


def main():
    args   = parse_args()
    params = json.load(open(args.config, encoding="utf-8"))

    if args.symbol:    params["symbol"]     = args.symbol
    if args.timeframe: params["timeframe"]  = args.timeframe
    if args.start_date: params["start_date"] = args.start_date
    if args.end_date:   params["end_date"]   = args.end_date

    symbol    = params["symbol"]
    timeframe = params["timeframe"]

    # 탐색 조합 생성
    targets = ["all"] if "all" in args.target else args.target
    if args.mode == "grid":
        combos = make_grid(targets)
    else:
        combos = make_random(targets, args.n_trials)

    # 파라미터 컬럼명 정리
    _groups = ["st", "rsi", "grad", "sl"] if "all" in targets else targets
    target_labels = ", ".join(
        k for g in _groups if g in SEARCH_SPACE for k in SEARCH_SPACE[g].keys()
    )

    print(f"\n{'='*65}")
    print(f"  파라미터 최적화  [{args.mode.upper()}]")
    print(f"  심볼: {symbol}  TF: {timeframe}")
    print(f"  기간: {params['start_date']} ~ {params['end_date']}")
    print(f"  대상: {', '.join(targets)}  조합수: {len(combos)}")
    print(f"{'='*65}\n")

    # 데이터 로드
    from data.fetcher import load_ohlcv_with_warmup
    df_raw = load_ohlcv_with_warmup(
        symbol        = symbol,
        timeframe     = timeframe,
        start_date    = params["start_date"],
        end_date      = params["end_date"],
        exchange_id   = params.get("exchange", "okx"),
        warmup_bars   = 200,
        force_refresh = args.refresh,
    )
    print(f"봉 수: {len(df_raw):,}\n")

    results = []
    for i, combo in enumerate(combos, 1):
        combo_str = "  ".join(f"{k}={v}" for k, v in combo.items())
        r = run_combo(df_raw, params, combo, args.min_trades)
        if r:
            results.append(r)
            if i % 10 == 0 or i <= 5:
                print(f"[{i:4d}/{len(combos)}] {combo_str}  "
                      f"→ {r['trades']}건 {r['win_rate']}% {r['profit']:+.0f}% PF{r['pf']}")
        else:
            if i % 50 == 0:
                print(f"[{i:4d}/{len(combos)}] {combo_str}  → 스킵")

    if not results:
        print("결과 없음.")
        return

    df_res = pd.DataFrame(results)
    sort_col = {
        "profit":  "profit",
        "pf":      "pf",
        "winrate": "win_rate",
        "trades":  "trades",
        "calmar":  "calmar",
    }[args.sort]
    df_res = df_res.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  상위 {min(args.top, len(df_res))}개 결과 (정렬: {args.sort})")
    print(f"  전체 유효 결과: {len(df_res)}개 / {len(combos)}개 시도")
    print(f"{'='*65}")
    print(df_res.head(args.top).to_string(index=False))

    # CSV 저장
    safe = f"{symbol.replace('/','_').replace(':','_')}_{timeframe}"
    out  = Path("results") / f"optimize_{safe}.csv"
    out.parent.mkdir(exist_ok=True)
    df_res.to_csv(out, index=False)
    print(f"\n  결과 저장: {out}")

    # 최적값 요약
    best = df_res.iloc[0]
    print(f"\n  ★ {args.sort} 기준 최적값:")
    for col in df_res.columns:
        print(f"    {col}: {best[col]}")


if __name__ == "__main__":
    main()
