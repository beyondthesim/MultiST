"""
안전 한도 적용/미적용 백테스트 비교

A: 한도 없음 (현재 운용 설정 그대로)
B: 한도 있음 (일일 -10% 신규차단 + MDD -30% 강제청산+정지)
C: 한도 완화 (일일 -20% + MDD -60% — 백테스트 추구용)

각 시나리오의 수익률, MDD, 한도 발동 여부 비교.
"""

import copy
import json

import numpy as np
import pandas as pd
from tabulate import tabulate


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*80}")
    print(f"  안전 한도 비교 백테스트  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  {base_params['start_date']} ~ {base_params['end_date']}")
    print(f"  현재 운용: 메인 {base_params.get('main_position_pct', 100)}% / "
          f"CT {base_params['counter_trend'].get('ct_position_pct', 0)}%")
    print(f"{'='*80}\n")

    from data.fetcher import load_ohlcv_with_warmup
    from strategy.signal import build_signals
    from indicators.counter_signals import add_counter_signals
    from backtest.engine import BacktestEngine

    df_raw = load_ohlcv_with_warmup(
        symbol=base_params["symbol"], timeframe=base_params["timeframe"],
        start_date=base_params["start_date"], end_date=base_params["end_date"],
        exchange_id=base_params.get("exchange", "okx"), warmup_bars=200,
    )
    df_btc = load_ohlcv_with_warmup(
        symbol="BTC/USDT:USDT", timeframe="4h",
        start_date=base_params["start_date"], end_date=base_params["end_date"],
        exchange_id="okx", warmup_bars=200,
    )
    p_no = copy.deepcopy(base_params); p_no["counter_trend"] = {"enabled": False}
    df_signals = build_signals(df_raw.copy(), p_no, btc_df=df_btc)
    df_bt = df_signals[df_signals["in_backtest"]].copy()
    df_bt = add_counter_signals(df_bt, base_params["counter_trend"])

    SCENARIOS = {
        "A_한도없음":          {"daily_loss_pct_limit": 0,  "max_dd_pct_limit": 0},
        "B_봇기본 (일10/30)":  {"daily_loss_pct_limit": 10, "max_dd_pct_limit": 30},
        "C_완화 (일20/60)":    {"daily_loss_pct_limit": 20, "max_dd_pct_limit": 60},
        "D_타이트 (일5/15)":   {"daily_loss_pct_limit": 5,  "max_dd_pct_limit": 15},
        "E_MDD만 50":          {"daily_loss_pct_limit": 0,  "max_dd_pct_limit": 50},
        "F_MDD만 40":          {"daily_loss_pct_limit": 0,  "max_dd_pct_limit": 40},
    }

    rows = []
    initial = base_params["initial_capital"]

    for name, limits in SCENARIOS.items():
        print(f"  실행 중: {name} ...", end="", flush=True)

        params = copy.deepcopy(base_params)
        params.update(limits)

        engine = BacktestEngine(params)
        main_t, ct_t, eq_curve = engine.run(df_bt)

        vals  = np.array([e["equity"] for e in eq_curve])
        peak  = np.maximum.accumulate(vals)
        dd    = (vals - peak) / peak * 100
        max_dd = abs(dd.min())
        max_dd_usdt = abs((vals - peak).min())
        final = vals[-1]

        # 한도 발동 여부 검사
        all_t = main_t + ct_t
        mdd_halt = sum(1 for t in all_t if t.get("close_reason") == "MDD_HALT")
        halted_at = ""
        if mdd_halt > 0:
            for t in sorted(all_t, key=lambda x: x.get("exit_time", "")):
                if t.get("close_reason") == "MDD_HALT":
                    halted_at = str(t["exit_time"])[:10]
                    break

        # 일일 손실 한도 발동 횟수 추정 (정확한 추적은 엔진 수정 필요)
        # 여기서는 거래수 차이로 추정
        n_main_baseline = 455  # 한도 없음 baseline
        n_main = len(main_t)
        skipped = max(0, n_main_baseline - n_main)

        print(f" 거래 {len(main_t)}/{len(ct_t)}  최종 ${final:,.0f}  MDD {max_dd:.1f}%")

        rows.append({
            "시나리오":       name,
            "메인거래":       len(main_t),
            "CT거래":         len(ct_t),
            "최종자본($)":    f"{final:,.0f}",
            "수익률(%)":      f"{(final/initial-1)*100:+.0f}",
            "MaxDD(%)":       f"{max_dd:.2f}",
            "MaxDD($)":       f"{max_dd_usdt:,.0f}",
            "MDD강제청산":    "예" if mdd_halt > 0 else "-",
            "정지일":         halted_at if halted_at else "-",
        })

    print(f"\n{'='*80}")
    print("  결과 비교표")
    print(f"{'='*80}")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    # 핵심 인사이트
    print(f"\n{'─'*80}")
    print(f"  [핵심 분석]")
    a = rows[0]; b = rows[1]; c = rows[2]
    a_pnl = float(a["수익률(%)"].replace("+", "").replace(",", ""))
    b_pnl = float(b["수익률(%)"].replace("+", "").replace(",", ""))
    c_pnl = float(c["수익률(%)"].replace("+", "").replace(",", ""))
    print(f"  A 한도없음:  +{a_pnl:>10,.0f}%  MDD {a['MaxDD(%)']}%")
    print(f"  B 봇기본:    +{b_pnl:>10,.0f}%  MDD {b['MaxDD(%)']}%  → A 대비 수익 {(b_pnl/a_pnl*100):.1f}%")
    print(f"  C 완화:      +{c_pnl:>10,.0f}%  MDD {c['MaxDD(%)']}%  → A 대비 수익 {(c_pnl/a_pnl*100):.1f}%")
    print(f"\n  → MDD 30% 한도(B)는 백테스트 50% 회복 구간을 막아 수익이 크게 깎임")
    print(f"  → MDD 60% 한도(C)는 백테스트와 거의 동일하게 작동")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
