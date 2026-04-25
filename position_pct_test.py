"""
%기반 동적 사이징 시나리오 비교

설정:
  - 메인: 자본 × 10% × 20레버리지 = 자본의 200% 명목 (main_position_pct=200)
  - CT 시나리오 A: 자본 × 30% × 20 = 자본의 600% (ct_position_pct=600)
  - CT 시나리오 B: 자본 × 20% × 20 = 자본의 400% (ct_position_pct=400)

매 진입 시 현재 자본의 % 만큼 진입 → 복리 효과로 자본 변할 때마다 사이즈 비례 변경.
"""

import copy
import json

import numpy as np
import pandas as pd


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*80}")
    print(f"  %기반 동적 사이징 시나리오 비교  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  {base_params['start_date']} ~ {base_params['end_date']}")
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

    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_signals = build_signals(df_raw.copy(), params_no_ct, btc_df=df_btc)
    df_bt_main_signals = df_signals[df_signals["in_backtest"]].copy()
    df_bt = add_counter_signals(df_bt_main_signals, base_params["counter_trend"])

    def _stats(name, eq_curve, main_t, ct_t, main_pos, ct_pos):
        if not eq_curve:
            return None
        vals = np.array([e["equity"] for e in eq_curve])
        peak = np.maximum.accumulate(vals)
        dd_pct  = (vals - peak) / peak * 100
        max_dd  = abs(dd_pct.min())
        max_dd_usdt = abs((vals - peak).min())
        final = vals[-1]

        def _wr(trades):
            if not trades: return 0.0, 0.0
            w = sum(1 for t in trades if t["net_pnl"] > 0)
            return w/len(trades)*100, sum(t["net_pnl"] for t in trades)

        m_wr, m_pnl = _wr(main_t)
        c_wr, c_pnl = _wr(ct_t)

        print(f"\n  ┌─ {name} ─────────────────────────────────────────")
        if isinstance(main_pos, (int, float)):
            print(f"  │ 메인: {main_pos}% 명목 (마진 {main_pos/20:.0f}% × 20레버리지)")
        else:
            print(f"  │ 메인: {main_pos}")
        if isinstance(ct_pos, (int, float)):
            print(f"  │ CT:   {ct_pos}% 명목 (마진 {ct_pos/20:.0f}% × 20레버리지)")
        else:
            print(f"  │ CT:   {ct_pos}")
        print(f"  │ 메인 거래: {len(main_t):>4}건  WR {m_wr:5.2f}%  PnL ${m_pnl:+10,.0f}")
        print(f"  │ CT 거래:   {len(ct_t):>4}건  WR {c_wr:5.2f}%  PnL ${c_pnl:+10,.0f}")
        print(f"  │ 최종자본: ${final:>12,.0f}  (수익률 {(final/1000-1)*100:+.0f}%)")
        print(f"  │ Max DD:   {max_dd:6.2f}%  (${max_dd_usdt:,.0f})")
        return {"final": final, "dd": max_dd, "dd_usdt": max_dd_usdt,
                "main_pnl": m_pnl, "ct_pnl": c_pnl,
                "n_main": len(main_t), "n_ct": len(ct_t),
                "main_wr": m_wr, "ct_wr": c_wr}

    initial = base_params["initial_capital"]

    # ── 시나리오 0: 비교 기준선 (현재 ct_size_pct=2000, 메인=자본100%) ────
    p0 = copy.deepcopy(base_params)
    p0.pop("main_position_pct", None)
    p0["counter_trend"] = {**p0["counter_trend"]}
    p0["counter_trend"].pop("ct_position_pct", None)
    e0 = BacktestEngine(p0)
    m0, c0, eq0 = e0.run(df_bt)
    s0 = _stats("[기준] 메인=자본100% / CT=ct_size_pct(고정)", eq0, m0, c0, 100, "고정")

    # ── 시나리오 A: 메인 200% / CT 600% ────────────────────────────
    pA = copy.deepcopy(base_params)
    pA["main_position_pct"] = 200
    pA["counter_trend"] = {**pA["counter_trend"], "ct_position_pct": 600}
    pA["counter_trend"].pop("ct_size_pct", None)
    eA = BacktestEngine(pA)
    mA, cA, eqA = eA.run(df_bt)
    sA = _stats("[A] 메인 10%×20 = 200% / CT 30%×20 = 600%", eqA, mA, cA, 200, 600)

    # ── 시나리오 B: 메인 200% / CT 400% ────────────────────────────
    pB = copy.deepcopy(base_params)
    pB["main_position_pct"] = 200
    pB["counter_trend"] = {**pB["counter_trend"], "ct_position_pct": 400}
    pB["counter_trend"].pop("ct_size_pct", None)
    eB = BacktestEngine(pB)
    mB, cB, eqB = eB.run(df_bt)
    sB = _stats("[B] 메인 10%×20 = 200% / CT 20%×20 = 400%", eqB, mB, cB, 200, 400)

    # ── 비교표 ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  [요약 비교표]")
    from tabulate import tabulate
    rows = []
    for name, s, mp, cp in [("기준 (현재)", s0, "100%", "고정"),
                              ("A (CT 30%)",  sA, "200%", "600%"),
                              ("B (CT 20%)",  sB, "200%", "400%")]:
        rows.append([
            name,
            mp, cp,
            f"${s['final']:,.0f}",
            f"{(s['final']/initial-1)*100:+.0f}%",
            f"{s['dd']:.2f}%",
            f"${s['dd_usdt']:,.0f}",
            f"${s['main_pnl']:+,.0f}",
            f"${s['ct_pnl']:+,.0f}",
        ])
    print(tabulate(rows, headers=["시나리오", "메인%", "CT%", "최종자본", "수익률", "MaxDD%", "MaxDD$", "메인PnL", "CT_PnL"], tablefmt="simple"))

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
