"""
멀티 심볼/타임프레임 스캔 - 최적 파라미터 탐색

사용법:
  python scan.py                          # 기본 심볼 목록 스캔
  python scan.py --symbols BTC ETH SOL   # 지정 심볼만
  python scan.py --tf 15m 1h             # 지정 타임프레임만
  python scan.py --sort pf               # 정렬 기준 변경
  python scan.py --top 10                # 상위 N개만 출력
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
    "NEAR/USDT:USDT",
    "LTC/USDT:USDT",
    "ATOM/USDT:USDT",
    "APT/USDT:USDT",
    "OP/USDT:USDT",
    "ARB/USDT:USDT",
    "INJ/USDT:USDT",
    "SUI/USDT:USDT",
    "STX/USDT:USDT",
    "FIL/USDT:USDT",
    "ETC/USDT:USDT",
    "ICP/USDT:USDT",
    "HBAR/USDT:USDT",
    "AAVE/USDT:USDT",
    "TON/USDT:USDT",
    "MATIC/USDT:USDT",
    "SEI/USDT:USDT",
    "TIA/USDT:USDT",
    "WLD/USDT:USDT",
    "PEPE/USDT:USDT",
    "WIF/USDT:USDT",
    "BONK/USDT:USDT",
    "JTO/USDT:USDT",
    "ENA/USDT:USDT",
    "RENDER/USDT:USDT",
    "TAO/USDT:USDT",
    "JUP/USDT:USDT",
    "PYTH/USDT:USDT",
    "BLUR/USDT:USDT",
]

DEFAULT_TFS = ["15m", "1h"]


def parse_args():
    p = argparse.ArgumentParser(description="Multi Supertrend 멀티 스캔")
    p.add_argument("--config",   default="config/params.json")
    p.add_argument("--symbols",  nargs="+", help="심볼 목록 (예: BTC ETH)")
    p.add_argument("--tf",       nargs="+", dest="timeframes", help="타임프레임 목록")
    p.add_argument("--start",    dest="start_date", help="시작일 YYYY-MM-DD")
    p.add_argument("--end",      dest="end_date",   help="종료일 YYYY-MM-DD")
    p.add_argument("--sort",     default="profit",
                   choices=["profit", "pf", "winrate", "trades"],
                   help="정렬 기준 (기본: profit)")
    p.add_argument("--top",      type=int, default=20, help="상위 N개 출력")
    p.add_argument("--refresh",  action="store_true", help="캐시 재수집")
    p.add_argument("--min-trades", type=int, default=30,
                   help="최소 거래수 필터 (기본: 30)")
    return p.parse_args()


def run_single(symbol: str, timeframe: str, params: dict,
               force_refresh: bool = False) -> dict | None:
    """단일 심볼/타임프레임 백테스트. 실패 시 None 반환."""
    from data.fetcher import load_ohlcv_with_warmup
    from strategy.signal import build_signals
    from backtest.engine import BacktestEngine

    p = dict(params)
    p["symbol"]    = symbol
    p["timeframe"] = timeframe

    try:
        df = load_ohlcv_with_warmup(
            symbol        = symbol,
            timeframe     = timeframe,
            start_date    = p["start_date"],
            end_date      = p["end_date"],
            exchange_id   = p.get("exchange", "okx"),
            warmup_bars   = 200,
            force_refresh = force_refresh,
        )
        df_s  = build_signals(df, p)
        df_bt = df_s[df_s["in_backtest"]].copy()

        engine = BacktestEngine(p)
        trades, _ct, eq_curve = engine.run(df_bt)

        if not trades:
            return None

        n        = len(trades)
        winners  = sum(1 for t in trades if t["net_pnl"] > 0)
        win_rate = winners / n * 100
        init_cap = p["initial_capital"]
        final_eq = eq_curve[-1]["equity"] if eq_curve else init_cap
        net_pnl  = final_eq - init_cap
        profit_pct = net_pnl / init_cap * 100

        gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        gross_loss   = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # MDD
        peak = init_cap
        mdd  = 0.0
        for pt in eq_curve:
            eq   = pt["equity"]
            peak = max(peak, eq)
            dd   = (peak - eq) / peak * 100
            mdd  = max(mdd, dd)

        return {
            "symbol":    symbol,
            "tf":        timeframe,
            "trades":    n,
            "win_rate":  round(win_rate, 1),
            "profit":    round(profit_pct, 1),
            "pf":        round(pf, 3),
            "mdd":       round(mdd, 1),
            "final_eq":  round(final_eq, 2),
        }

    except Exception as e:
        print(f"  [{symbol} {timeframe}] 오류: {e}")
        return None


def main():
    args = parse_args()
    params = json.load(open(args.config, encoding="utf-8"))

    symbols    = args.symbols or DEFAULT_SYMBOLS
    timeframes = args.timeframes or DEFAULT_TFS

    if args.start_date:
        params["start_date"] = args.start_date
    if args.end_date:
        params["end_date"] = args.end_date

    total = len(symbols) * len(timeframes)
    print(f"\n{'='*60}")
    print(f"  Multi Supertrend 스캔")
    print(f"  기간: {params['start_date']} ~ {params['end_date']}")
    print(f"  심볼 {len(symbols)}개 × TF {len(timeframes)}개 = 총 {total}개 조합")
    print(f"{'='*60}\n")

    results = []
    done = 0
    for sym in symbols:
        # 심볼 형식 정규화 (BTC → BTC/USDT:USDT)
        if "/" not in sym:
            sym = f"{sym}/USDT:USDT"

        for tf in timeframes:
            done += 1
            print(f"[{done:3d}/{total}] {sym} {tf}  ", end="", flush=True)
            r = run_single(sym, tf, params, args.refresh)
            if r:
                print(f"→ 거래:{r['trades']}건  승률:{r['win_rate']}%  수익:{r['profit']:+.1f}%  PF:{r['pf']}")
                if r["trades"] >= args.min_trades:
                    results.append(r)
            else:
                print("→ 실패")

    if not results:
        print("\n결과 없음.")
        return

    df_res = pd.DataFrame(results)

    sort_col = {
        "profit":  "profit",
        "pf":      "pf",
        "winrate": "win_rate",
        "trades":  "trades",
    }[args.sort]

    df_res = df_res.sort_values(sort_col, ascending=False).head(args.top)

    print(f"\n{'='*60}")
    print(f"  상위 {len(df_res)}개 결과 (정렬: {args.sort}, 최소 거래수: {args.min_trades})")
    print(f"{'='*60}")
    print(df_res.to_string(index=False))

    # CSV 저장
    out = Path("results") / "scan_results.csv"
    out.parent.mkdir(exist_ok=True)
    df_res.to_csv(out, index=False)
    print(f"\n  결과 저장: {out}")


if __name__ == "__main__":
    main()
