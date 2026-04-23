"""
주중(월~금) 진입 제한 vs 전체 시간 진입 비교 백테스트

사용법:
  python weekday_compare.py
  python weekday_compare.py --config config/params.json
"""

import argparse
import copy
import json

from tabulate import tabulate


def load_params(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_backtest(params: dict, weekday_only: bool):
    from data.fetcher import load_ohlcv_with_warmup
    from strategy.signal import build_signals
    from backtest.engine import BacktestEngine

    df = load_ohlcv_with_warmup(
        symbol       = params["symbol"],
        timeframe    = params["timeframe"],
        start_date   = params["start_date"],
        end_date     = params["end_date"],
        exchange_id  = params.get("exchange", "okx"),
        warmup_bars  = 200,
        force_refresh= False,
    )

    df = build_signals(df, params)
    df_bt = df[df["in_backtest"]].copy()

    p = copy.deepcopy(params)
    p["weekday_only"] = weekday_only

    engine = BacktestEngine(p)
    main_trades, _ct_trades, equity_curve = engine.run(df_bt)
    return main_trades, equity_curve


def compute_metrics(trades, equity_curve, initial_capital):
    from backtest.reporter import compute_metrics
    return compute_metrics(trades, equity_curve, initial_capital)


def print_comparison(m_all: dict, m_wd: dict, initial: float) -> None:
    def fmt_pf(v):
        return f"{v:.3f}" if v != float("inf") else "∞"

    rows = [
        ["총 거래수",      m_all["total_trades"],                        m_wd["total_trades"]],
        ["수익 거래",      f"{m_all['winning_trades']} ({m_all['win_rate']:.1f}%)",
                           f"{m_wd['winning_trades']} ({m_wd['win_rate']:.1f}%)"],
        ["손실 거래",      m_all["losing_trades"],                       m_wd["losing_trades"]],
        ["총 손익 (%)",    f"{m_all['net_profit_pct']:+.2f}%",           f"{m_wd['net_profit_pct']:+.2f}%"],
        ["총 손익 (USDT)", f"{m_all['net_profit_usdt']:+,.2f}",          f"{m_wd['net_profit_usdt']:+,.2f}"],
        ["최종 자산",      f"{m_all['final_equity']:,.2f}",              f"{m_wd['final_equity']:,.2f}"],
        ["최대 DD (%)",    f"{m_all['max_drawdown_pct']:.2f}%",          f"{m_wd['max_drawdown_pct']:.2f}%"],
        ["최대 DD (USDT)", f"{m_all['max_drawdown_usdt']:,.2f}",         f"{m_wd['max_drawdown_usdt']:,.2f}"],
        ["수익지수 (PF)",  fmt_pf(m_all["profit_factor"]),               fmt_pf(m_wd["profit_factor"])],
        ["평균 수익",      f"{m_all['avg_win']:+,.2f}",                  f"{m_wd['avg_win']:+,.2f}"],
        ["평균 손실",      f"{m_all['avg_loss']:+,.2f}",                 f"{m_wd['avg_loss']:+,.2f}"],
        ["최대 연속 승",   m_all["max_consec_win"],                      m_wd["max_consec_win"]],
        ["최대 연속 패",   m_all["max_consec_loss"],                     m_wd["max_consec_loss"]],
    ]

    print()
    print("=" * 65)
    print("  진입 필터 비교: 전체 시간 vs 주중(월~금)만")
    print("=" * 65)
    print(tabulate(rows, headers=["지표", "전체 시간", "주중(월~금)"], tablefmt="simple"))
    print("=" * 65)

    # 개선율 계산
    def delta(a, b, higher_is_better=True):
        if a == 0:
            return "-"
        d = (b - a) / abs(a) * 100
        sign = "+" if d >= 0 else ""
        flag = "▲" if (d > 0) == higher_is_better else "▼"
        return f"{flag} {sign}{d:.1f}%"

    print()
    print("  [주중 필터 적용 시 변화]")
    print(f"  거래수    : {delta(m_all['total_trades'],      m_wd['total_trades'],      False)}")
    print(f"  총 손익   : {delta(m_all['net_profit_pct'],    m_wd['net_profit_pct'],    True)}")
    print(f"  최대 DD   : {delta(m_all['max_drawdown_pct'],  m_wd['max_drawdown_pct'],  False)}")
    print(f"  수익지수  : {delta(m_all['profit_factor'],     m_wd['profit_factor'],     True)}")
    print(f"  승률      : {delta(m_all['win_rate'],          m_wd['win_rate'],          True)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.json")
    args = parser.parse_args()

    params = load_params(args.config)

    print(f"\n{'='*65}")
    print(f"  주중 필터 비교 백테스트")
    print(f"  심볼: {params['symbol']}  타임프레임: {params['timeframe']}")
    print(f"  기간: {params['start_date']} ~ {params['end_date']}")
    print(f"{'='*65}")

    print("\n[1] 데이터 수집 및 전체 시간 백테스트...")
    trades_all, eq_all = run_backtest(params, weekday_only=False)
    print(f"  전체 시간: {len(trades_all)}건")

    print("\n[2] 주중(월~금) 진입 제한 백테스트...")
    trades_wd, eq_wd = run_backtest(params, weekday_only=True)
    print(f"  주중 전용: {len(trades_wd)}건")

    initial = params["initial_capital"]
    m_all = compute_metrics(trades_all, eq_all, initial)
    m_wd  = compute_metrics(trades_wd,  eq_wd,  initial)

    print_comparison(m_all, m_wd, initial)

    # 주말 진입 거래만 별도 통계
    wd_entry_times = {t["entry_time"] for t in trades_wd}
    weekend_trades = [
        t for t in trades_all
        if t["entry_time"] not in wd_entry_times
    ]
    if weekend_trades:
        import pandas as pd
        df_we = pd.DataFrame(weekend_trades)
        we_wins = df_we["is_winner"].sum()
        we_total = len(df_we)
        we_pnl = df_we["net_pnl"].sum()
        print(f"\n  [주말 진입 거래만] {we_total}건 | 승률 {we_wins/we_total*100:.1f}% | 합산 PnL {we_pnl:+,.2f} USDT")
    else:
        print("\n  [주말 진입 거래] 없음")


if __name__ == "__main__":
    main()
