"""
C2_FIB x20 Max DD 검증

의심: 사이즈 20배인데 Max DD가 26%밖에 안 된다는 게 이상함.
검증 방법:
  1. equity curve를 추적하여 Max DD 발생 시점 식별
  2. 그 시점 부근의 CT 거래 확인 (CT가 cushion 역할인지, 아니면 운 좋게 비켜갔는지)
  3. CT 단독 자본 곡선의 DD vs 합산 DD 비교
  4. 만약 CT SL이 메인 DD 시점과 겹쳤다면 어땠을지 stress test
"""

import copy
import json

import numpy as np
import pandas as pd


def _fib(n: int) -> list[int]:
    out = [1, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


C2_FIB = {
    "enabled": True, "equity_pct": 0, "ct_size_pct": 2000,
    "rsi_period": 14, "consec_candles": 4,
    "consec_candles_long": 5, "consec_candles_short": 4,
    "min_candle_pct": 0.0,
    "ema_filter": {"enabled": True, "length": 30},
    "divergence_pivot_period": 5, "divergence_max_bars": 60,
    "ct_long_enabled": True, "ct_short_enabled": True,
    "rsi_long_entry1": 25, "rsi_short_entry1": 65,
    "rsi_long_exit": 65, "rsi_short_exit": 35,
    "ct_exit_enabled": False,
    "max_dca": 10, "dca_weights": _fib(10),
    "dca_price_pct": 0.015, "dca_require_divergence": False,
    "sl_long_pct": 0.10, "sl_short_pct": 0.10,
    "ct_long_tp":  [{"pct": 0.03, "qty_pct": 100}],
    "ct_short_tp": [{"pct": 0.03, "qty_pct": 100}],
    "all_close_pct": 0.10, "safe_close_count": 2, "safe_close_pct": 0.02,
    "ct_bottom_quality_min": 2,
    "ct_bottom_quality_min_long": 3, "ct_bottom_quality_min_short": 2,
}


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*82}")
    print(f"  C2_FIB x20 Max DD 검증")
    print(f"{'='*82}\n")

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
    df_bt = df_signals[df_signals["in_backtest"]].copy()
    df_bt = add_counter_signals(df_bt, C2_FIB)

    # 시나리오 1: 메인만 (CT 비활성화)
    p1 = copy.deepcopy(base_params); p1["counter_trend"] = {"enabled": False}
    eq1 = BacktestEngine(p1).run(df_bt)[2]

    # 시나리오 2: 메인 + C2_FIB x20
    p2 = copy.deepcopy(base_params); p2["counter_trend"] = C2_FIB
    main_t, ct_t, eq2 = BacktestEngine(p2).run(df_bt)

    # 시나리오 3: 메인 + BASE_DCA0 (현재 운용 설정)
    p3 = copy.deepcopy(base_params)
    p3["counter_trend"] = {**C2_FIB,
                            "max_dca": 0, "dca_weights": [],
                            "ct_size_pct": 100,
                            "sl_long_pct": 0.015, "sl_short_pct": 0.015,
                            "rsi_long_entry1": 25, "rsi_short_entry1": 65,
                            "ct_long_tp":  [{"pct": 0.02, "qty_pct": 80}, {"pct": 0.04, "qty_pct": 100}],
                            "ct_short_tp": [{"pct": 0.07, "qty_pct": 50}, {"pct": 0.12, "qty_pct": 100}]}
    eq3 = BacktestEngine(p3).run(df_bt)[2]

    def _max_dd(eq_curve):
        vals = np.array([e["equity"] for e in eq_curve])
        ts   = [e["timestamp"] for e in eq_curve]
        peak = np.maximum.accumulate(vals)
        dd   = (vals - peak) / peak * 100
        idx_min = int(np.argmin(dd))
        idx_peak = int(np.argmax(vals[:idx_min+1]))
        return {
            "max_dd_pct":  abs(dd.min()),
            "max_dd_usdt": abs((vals - peak).min()),
            "peak_ts":     ts[idx_peak],
            "trough_ts":   ts[idx_min],
            "peak_eq":     vals[idx_peak],
            "trough_eq":   vals[idx_min],
            "final_eq":    vals[-1],
        }

    def _stats(name, eq_curve):
        m = _max_dd(eq_curve)
        print(f"\n  [{name}]")
        print(f"    최종자본: ${m['final_eq']:,.0f}  (수익률 {(m['final_eq']/1000-1)*100:+.0f}%)")
        print(f"    Max DD:   {m['max_dd_pct']:.2f}%  (${m['max_dd_usdt']:,.0f})")
        print(f"    DD 기간:  {m['peak_ts']} (peak ${m['peak_eq']:,.0f})")
        print(f"           →  {m['trough_ts']} (trough ${m['trough_eq']:,.0f})")
        return m

    m1 = _stats("시나리오 1: 메인만 (CT 비활성)",                eq1)
    m2 = _stats("시나리오 2: 메인 + C2_FIB x20 (요청)",          eq2)
    m3 = _stats("시나리오 3: 메인 + BASE_DCA0 (현재 운용)",      eq3)

    # CT 단독 누적 P&L 시계열 (시나리오 2 기준)
    print(f"\n{'─'*82}")
    print(f"  [CT x20 거래 분석]")
    print(f"  CT 거래수: {len(ct_t)}건")
    if ct_t:
        ct_df = pd.DataFrame(ct_t)
        ct_df["exit_time"] = pd.to_datetime(ct_df["exit_time"])
        ct_df = ct_df.sort_values("exit_time")
        ct_df["cumulative"] = ct_df["net_pnl"].cumsum()
        ct_peak = ct_df["cumulative"].cummax()
        ct_dd   = (ct_df["cumulative"] - ct_peak)
        ct_dd_min = ct_dd.min()
        ct_max_dd_pct = abs(ct_dd_min / 20000.0 * 100)  # ct_base $20,000 대비

        print(f"  CT 누적 P&L 최저점: ${ct_dd_min:,.1f} (ct_base $20,000 대비 {ct_max_dd_pct:.2f}%)")
        print(f"  CT SL 발생 시점:")
        sl_trades = ct_df[ct_df["close_reason"] == "SL"]
        for _, t in sl_trades.iterrows():
            print(f"    {t['exit_time']}  {t['direction']}  net_pnl=${t['net_pnl']:+.1f}")

        # 메인 DD 시점과 CT 거래 시점 겹침 분석
        print(f"\n  [메인 DD 시점 부근 ({m1['peak_ts']} → {m1['trough_ts']}) CT 활동]")
        peak_dt = pd.to_datetime(m1["peak_ts"])
        trough_dt = pd.to_datetime(m1["trough_ts"])
        ct_during = ct_df[(ct_df["exit_time"] >= peak_dt) & (ct_df["exit_time"] <= trough_dt)]
        if len(ct_during):
            tp_count = (ct_during["close_reason"] == "TP").sum() + (ct_during["close_reason"] == "TP_FULL").sum()
            sl_count = (ct_during["close_reason"] == "SL").sum()
            ct_pnl_during = ct_during["net_pnl"].sum()
            print(f"    구간 내 CT 거래: {len(ct_during)}건 (TP {tp_count} / SL {sl_count})")
            print(f"    구간 내 CT 누적 P&L: ${ct_pnl_during:+.1f}")
            print(f"    → CT가 메인 DD를 ${ct_pnl_during:+.0f} 만큼 상쇄/악화")
        else:
            print(f"    구간 내 CT 거래 없음")

    # Stress test: 만약 CT SL이 추가로 발생했다면?
    print(f"\n{'─'*82}")
    print(f"  [STRESS TEST: 가정 - 추가 CT SL 발생 시]")
    base_ct_pnl = sum(t["net_pnl"] for t in ct_t)
    base_total = m2["final_eq"] - 1000
    main_only_pnl = m1["final_eq"] - 1000
    print(f"  현실: CT 순익 ${base_ct_pnl:+,.0f}, 합산 ${base_total:+,.0f} (+{base_total/10:.0f}%)")
    print(f"  메인 단독: ${main_only_pnl:+,.0f} (+{main_only_pnl/10:.0f}%)")
    print()
    sl_loss_per = -42  # SL 1회 평균 손실 (x20 기준)
    for n_extra in [1, 3, 5, 10, 20]:
        new_ct = base_ct_pnl + n_extra * sl_loss_per
        new_total = main_only_pnl + new_ct
        diff_vs_base = new_total - base_total
        print(f"  +{n_extra:2d}회 SL 추가 시:  CT ${new_ct:+,.0f}  /  합산 ${new_total:+,.0f}  "
              f"(현실 대비 {diff_vs_base:+,.0f})")

    print(f"{'='*82}\n")


if __name__ == "__main__":
    main()
