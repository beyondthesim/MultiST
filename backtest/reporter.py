"""
백테스트 성과 리포트 생성
트레이딩뷰 '전략 리포트' 스타일 출력
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate


def compute_metrics(trades: list[dict], equity_curve: list[dict], initial_capital: float) -> dict:
    """핵심 성과 지표 계산"""
    if not trades:
        return {}

    df_trades = pd.DataFrame(trades)
    eq_df     = pd.DataFrame(equity_curve).set_index("timestamp")

    total_trades  = len(df_trades)
    winners       = df_trades[df_trades["is_winner"]]
    losers        = df_trades[~df_trades["is_winner"]]

    win_rate = len(winners) / total_trades * 100

    gross_profit = winners["net_pnl"].sum() if len(winners) > 0 else 0
    gross_loss   = abs(losers["net_pnl"].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_equity = eq_df["equity"].iloc[-1]
    net_profit_pct = (final_equity / initial_capital - 1) * 100

    # 최대 자본 감소 (Drawdown)
    equity_vals = eq_df["equity"].values
    peak        = np.maximum.accumulate(equity_vals)
    drawdown    = (equity_vals - peak) / peak * 100
    max_dd_pct  = abs(drawdown.min())
    max_dd_usdt = abs((equity_vals - peak).min())

    # 평균 수익/손실
    avg_win  = winners["net_pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["net_pnl"].mean()  if len(losers)  > 0 else 0

    # 연속 승/패
    results = df_trades["is_winner"].values
    max_consec_win  = _max_consecutive(results, True)
    max_consec_loss = _max_consecutive(results, False)

    return {
        "total_trades":      total_trades,
        "winning_trades":    len(winners),
        "losing_trades":     len(losers),
        "win_rate":          win_rate,
        "gross_profit":      gross_profit,
        "gross_loss":        gross_loss,
        "profit_factor":     profit_factor,
        "net_profit_usdt":   final_equity - initial_capital,
        "net_profit_pct":    net_profit_pct,
        "final_equity":      final_equity,
        "initial_capital":   initial_capital,
        "max_drawdown_pct":  max_dd_pct,
        "max_drawdown_usdt": max_dd_usdt,
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "max_consec_win":    max_consec_win,
        "max_consec_loss":   max_consec_loss,
    }


def _max_consecutive(arr: np.ndarray, value: bool) -> int:
    max_count = 0
    count = 0
    for v in arr:
        if v == value:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def monthly_returns(trades: list[dict], initial_capital: float) -> pd.DataFrame:
    """월별 수익률 계산 (트레이딩뷰 스타일)"""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"]  = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month

    # 누적 equity 계산 (연도별 월별 리턴%)
    equity = initial_capital
    monthly: dict = defaultdict(dict)

    for _, row in df.iterrows():
        year  = row["year"]
        month = row["month"]
        pnl   = row["net_pnl"]

        prev_eq = equity
        equity  = equity + pnl

        ret_pct = (equity / prev_eq - 1) * 100
        key = (year, month)
        monthly[key] = monthly.get(key, 0) + ret_pct

    years  = sorted({k[0] for k in monthly})
    months = list(range(1, 13))
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec","Year"]

    rows = []
    for year in years:
        row_data = [year]
        year_compound = 1.0
        for m in months:
            ret = monthly.get((year, m), None)
            if ret is not None:
                row_data.append(f"{ret:+.1f}")
                year_compound *= (1 + ret / 100)
            else:
                row_data.append("-")
        year_pct = (year_compound - 1) * 100
        row_data.append(f"{year_pct:+.1f}")
        rows.append(row_data)

    return pd.DataFrame(rows, columns=["Year"] + month_abbr)


def compute_ct_metrics(ct_trades: list[dict]) -> dict:
    """역추세 전략 전용 지표 (equity curve 없이 거래 목록만 사용)"""
    if not ct_trades:
        return {}

    df = pd.DataFrame(ct_trades)
    winners = df[df["is_winner"]]
    losers  = df[~df["is_winner"]]

    gp = winners["net_pnl"].sum() if len(winners) > 0 else 0.0
    gl = abs(losers["net_pnl"].sum()) if len(losers) > 0 else 0.0

    dca_col = df["dca_count"] if "dca_count" in df.columns else pd.Series([0])

    return {
        "total_trades":   len(df),
        "winning_trades": len(winners),
        "losing_trades":  len(losers),
        "win_rate":       len(winners) / len(df) * 100,
        "net_pnl":        df["net_pnl"].sum(),
        "profit_factor":  gp / gl if gl > 0 else float("inf"),
        "avg_win":        winners["net_pnl"].mean() if len(winners) > 0 else 0.0,
        "avg_loss":       losers["net_pnl"].mean()  if len(losers)  > 0 else 0.0,
        "avg_dca":        dca_col.mean(),
        "max_dca":        dca_col.max(),
        "dca_trades":     (dca_col > 0).sum(),
    }


def print_report(
    main_trades: list[dict],
    equity_curve: list[dict],
    params: dict,
    title: str = "백테스트 결과",
    ct_trades: Optional[list[dict]] = None,
) -> None:
    """전체 성과 리포트 출력 (메인 + 역추세 분리)"""
    ct_trades = ct_trades or []
    initial   = params["initial_capital"]

    ct_cfg     = params.get("counter_trend", {})
    ct_pct     = ct_cfg.get("equity_pct", 0) if ct_cfg.get("enabled") else 0
    leveraged  = ct_cfg.get("enabled", False) and ct_pct == 0
    ct_size_pct = ct_cfg.get("ct_size_pct", 100) / 100.0 if ct_cfg.get("enabled") else 1.0
    main_init  = initial
    ct_init    = initial * ct_size_pct if leveraged else initial * (ct_pct / 100.0)

    m = compute_metrics(main_trades, equity_curve, initial)

    if not m:
        print("거래 없음.")
        return

    print()
    print("=" * 60)
    print(f"  {title}")
    print(f"  {params.get('symbol','')} | {params.get('timeframe','')} | "
          f"{params.get('start_date','')} ~ {params.get('end_date','')}")
    print("=" * 60)

    summary = [
        ["초기 자본 (합산)",  f"{initial:,.2f} USDT  "
                               + (f"(레버리지: 메인 {initial:,.0f} / CT {ct_init:,.0f}={ct_size_pct*100:.0f}%)" if leveraged
                                  else f"(메인 {main_init:,.0f} + CT {ct_init:,.0f})")],
        ["최종 자산 (합산)",  f"{m['final_equity']:,.2f} USDT"],
        ["총 손익",           f"{m['net_profit_usdt']:+,.2f} USDT  ({m['net_profit_pct']:+.2f}%)"],
        ["최대 자본 감소",    f"{m['max_drawdown_usdt']:,.2f} USDT  ({m['max_drawdown_pct']:.2f}%)"],
        ["총 거래수 (메인)",  f"{m['total_trades']}"],
        ["수익 거래",         f"{m['winning_trades']} ({m['win_rate']:.2f}%)"],
        ["손실 거래",         f"{m['losing_trades']}"],
        ["수익지수(PF)",      f"{m['profit_factor']:.3f}"],
        ["평균 수익",         f"{m['avg_win']:+,.2f} USDT"],
        ["평균 손실",         f"{m['avg_loss']:+,.2f} USDT"],
        ["최대 연속 승",      f"{m['max_consec_win']}"],
        ["최대 연속 패",      f"{m['max_consec_loss']}"],
    ]
    print(tabulate(summary, tablefmt="simple"))

    # 역추세 전략 별도 성과
    if ct_trades:
        ctm = compute_ct_metrics(ct_trades)
        print()
        print("── 역추세(CT) 전략 성과 ──")
        ct_summary = [
            ["CT 초기 자본",  f"{ct_init:,.2f} USDT"],
            ["CT 누적 손익",  f"{ctm['net_pnl']:+,.2f} USDT"],
            ["CT 거래수",     f"{ctm['total_trades']}"],
            ["CT 승률",       f"{ctm['win_rate']:.2f}%"],
            ["CT 수익지수",   f"{ctm['profit_factor']:.3f}"],
            ["CT 평균 수익",  f"{ctm['avg_win']:+,.2f} USDT"],
            ["CT 평균 손실",  f"{ctm['avg_loss']:+,.2f} USDT"],
            ["DCA 실행 건수", f"{ctm['dca_trades']} / {ctm['total_trades']}"],
            ["평균 DCA 횟수", f"{ctm['avg_dca']:.2f}"],
            ["최대 DCA 횟수", f"{ctm['max_dca']:.0f}"],
        ]
        print(tabulate(ct_summary, tablefmt="simple"))

        # CT 청산 사유 분포
        df_ct = pd.DataFrame(ct_trades)
        if "close_reason" in df_ct.columns:
            reason_counts = df_ct["close_reason"].value_counts()
            print()
            print("── CT 청산 사유 ──")
            for reason, cnt in reason_counts.items():
                print(f"  {reason}: {cnt}건")

    # 월별 수익률
    mret = monthly_returns(main_trades, initial)
    if not mret.empty:
        print()
        print("── 월별 수익률 (%) [메인 기준] ──")
        print(tabulate(mret, headers="keys", tablefmt="simple", showindex=False))

    # 최근 메인 거래 10건
    if main_trades:
        df_t = pd.DataFrame(main_trades)
        cols = ["direction", "entry_time", "exit_time",
                "entry_price", "avg_exit_price", "net_pnl", "net_pnl_pct", "close_reason"]
        cols = [c for c in cols if c in df_t.columns]
        df_show = df_t[cols].tail(10).copy()
        if "net_pnl" in df_show.columns:
            df_show["net_pnl"] = df_show["net_pnl"].map("{:+.2f}".format)
        if "net_pnl_pct" in df_show.columns:
            df_show["net_pnl_pct"] = df_show["net_pnl_pct"].map("{:+.2f}%".format)
        print()
        print("── 최근 메인 거래 10건 ──")
        print(tabulate(df_show, headers="keys", tablefmt="simple", showindex=False))

    print("=" * 60)
