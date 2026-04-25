"""
메인 LONG / 메인 SHORT / CT LONG / CT SHORT 4방향 성과 비교

results/*.csv 파일을 읽어서 각 방향별 독립 성과를 출력.
main.py를 먼저 실행해 거래 내역이 저장되어 있어야 함.

사용법:
  python compare_directions.py
  python compare_directions.py --symbol ETH/USDT:USDT --tf 15m --start 2025-01-01
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: 0 for k in ["n", "wr", "net", "pf", "avg_w", "avg_l",
                               "max_w", "max_l", "tp", "sl", "st_flip"]}
    wins   = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    gp = wins["net_pnl"].sum()
    gl = abs(losses["net_pnl"].sum())
    reasons = df["close_reason"].value_counts() if "close_reason" in df.columns else pd.Series()

    return {
        "n":       len(df),
        "wr":      len(wins) / len(df) * 100 if len(df) else 0,
        "net":     df["net_pnl"].sum(),
        "pf":      gp / gl if gl > 0 else float("inf"),
        "avg_w":   wins["net_pnl"].mean()   if len(wins)   else 0,
        "avg_l":   losses["net_pnl"].mean() if len(losses) else 0,
        "max_w":   wins["net_pnl"].max()    if len(wins)   else 0,
        "max_l":   losses["net_pnl"].min()  if len(losses) else 0,
        "tp":      int(reasons.get("TP", 0) + reasons.get("TP_FULL", 0)),
        "sl":      int(reasons.get("SL", 0)),
        "st_flip": int(reasons.get("ST_FLIP", 0)),
    }


def _fmt_row(name: str, m: dict) -> dict:
    rr = abs(m["avg_w"] / m["avg_l"]) if m["avg_l"] != 0 else 0
    return {
        "방향":       name,
        "거래수":     m["n"],
        "승률(%)":    f"{m['wr']:.2f}",
        "순익(USDT)": f"{m['net']:+.2f}",
        "PF":         f"{m['pf']:.3f}" if m["pf"] != float("inf") else "∞",
        "R/R":        f"{rr:.2f}",
        "평균수익":   f"{m['avg_w']:+.2f}",
        "평균손실":   f"{m['avg_l']:+.2f}",
        "최대수익":   f"{m['max_w']:+.2f}",
        "최대손실":   f"{m['max_l']:+.2f}",
        "TP/SL/FLIP": f"{m['tp']}/{m['sl']}/{m['st_flip']}",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/params.json")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        params = json.load(f)

    safe_sym = params["symbol"].replace("/", "_").replace(":", "_")
    main_path = Path("results") / f"{safe_sym}_{params['timeframe']}_{params['start_date']}.csv"
    ct_path   = Path("results") / f"{safe_sym}_{params['timeframe']}_{params['start_date']}_ct.csv"

    if not main_path.exists():
        print(f"[경고] 메인 거래 CSV가 없음: {main_path}")
        print("먼저 python main.py 를 실행하세요.")
        return

    main_df = pd.read_csv(main_path)
    ct_df   = pd.read_csv(ct_path) if ct_path.exists() else pd.DataFrame()

    print(f"\n{'='*80}")
    print(f"  4방향 성과 비교  |  {params['symbol']}  {params['timeframe']}")
    print(f"  {params['start_date']} ~ {params['end_date']}")
    print(f"{'='*80}\n")

    main_long  = main_df[main_df["direction"] == "LONG"]
    main_short = main_df[main_df["direction"] == "SHORT"]
    ct_long    = ct_df[ct_df["direction"] == "LONG"]   if not ct_df.empty else pd.DataFrame()
    ct_short   = ct_df[ct_df["direction"] == "SHORT"]  if not ct_df.empty else pd.DataFrame()

    rows = [
        _fmt_row("메인 LONG",  _metrics(main_long)),
        _fmt_row("메인 SHORT", _metrics(main_short)),
        _fmt_row("CT LONG",    _metrics(ct_long)),
        _fmt_row("CT SHORT",   _metrics(ct_short)),
    ]

    # 합산 행
    total_main = _metrics(main_df)
    total_ct   = _metrics(ct_df) if not ct_df.empty else _metrics(pd.DataFrame())
    total_all  = _metrics(pd.concat([main_df, ct_df], ignore_index=True)) \
                 if not ct_df.empty else total_main

    rows.append({k: "─" * 10 for k in rows[0]})  # 구분선
    rows.append(_fmt_row("메인 전체", total_main))
    rows.append(_fmt_row("CT 전체",   total_ct))
    rows.append(_fmt_row("**전체**",  total_all))

    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))

    # 비중 분석
    initial = params["initial_capital"]
    print(f"\n{'─'*80}")
    print("  [비중 분석]")
    total_net = total_all["net"]
    if total_net != 0:
        for name, m in [("메인 LONG",  _metrics(main_long)),
                        ("메인 SHORT", _metrics(main_short)),
                        ("CT LONG",    _metrics(ct_long)),
                        ("CT SHORT",   _metrics(ct_short))]:
            share = m["net"] / total_net * 100
            bar_len = int(abs(share) / 2)
            bar = ("+" if share >= 0 else "-") * min(bar_len, 40)
            print(f"  {name:11s}: {m['net']:+8.1f} USDT  ({share:+6.1f}%)  {bar}")

    print(f"\n  초기자본: {initial:,.0f} USDT")
    print(f"  총 순익:  {total_net:+,.2f} USDT  ({total_net/initial*100:+.2f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
