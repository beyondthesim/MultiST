"""
CT LONG 단독 최적화 (베어장 데드캣 바운스 수확 전략)

베어장에서 CT LONG의 본질:
  - 큰 반등은 드물다 → TP가 크면 도달 못 하고 SL
  - 짧은 데드캣 바운스 (1-3%) 노리고 빠른 회수
  - 진입 자체를 까다롭게: capitulation 시점만
  - DCA로 평균가 낮춰 작은 반등으로 익절

테스트 차원:
  A. TP 크기      (4%/8% → 1.5%/3%)
  B. 진입 RSI     (30 → 25 → 22 → 20)
  C. 연속 음봉    (4 → 5 → 6)
  D. quality_min  (2 → 3 — 와이크+볼륨+바디수축 모두)
  E. SL 타이트    (1.5% → 1.0%)
  F. DCA 1회      (평균가 낮춰 빠른 익절)
  G. 복합 정밀    (모든 조건 결합)

사용법: python ct_long_optimize.py
"""

import copy
import json

from tabulate import tabulate


# ── 기준선 CT LONG 파라미터 (CT SHORT 비활성, CT LONG만 활성) ───────────
BASE_CT = {
    "enabled":               True,
    "equity_pct":            0,
    "rsi_period":            14,
    "consec_candles":        4,
    "min_candle_pct":        0.0,
    "ema_filter":            {"enabled": True, "length": 30},
    "divergence_pivot_period": 5,
    "divergence_max_bars":   60,

    "ct_long_enabled":       True,    # ★ CT LONG 활성
    "ct_short_enabled":      False,   # ★ CT SHORT 비활성

    "rsi_long_entry1":       30,
    "rsi_short_entry1":      65,
    "rsi_long_exit":         65,
    "rsi_short_exit":        35,
    "ct_exit_enabled":       False,
    "max_dca":               0,
    "dca_weights":           [],
    "dca_price_pct":         0.015,
    "dca_require_divergence": False,
    "sl_long_pct":           0.015,
    "sl_short_pct":          0.015,
    "all_close_pct":         0.10,
    "safe_close_count":      2,
    "safe_close_pct":        0.02,
    "ct_bottom_quality_min": 2,

    "ct_long_tp":  [{"pct": 0.07, "qty_pct": 50}, {"pct": 0.12, "qty_pct": 100}],
    "ct_short_tp": [{"pct": 0.07, "qty_pct": 50}, {"pct": 0.12, "qty_pct": 100}],
}


# ── 라운드 3: 베스트 조합들 결합 (+$100 돌파 시도) ───────────────────
# 라운드 2 베스트: I1_TP2(70)+4 +$62 / J2_CONSEC5 +$52 / L1_SL1_RSI22 +$45
# 핵심: quality3 + 1단계 TP 비중 ↑ + SL 타이트 + 진입 강화 결합
_R2_BEST = {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25}

LONG_VARIANTS_OLD = {
    # [라운드 1 우승]
    "D1_BASE":             {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── H. SL 타이트 (1.5% → 1.0%) → R/R 개선 ────────────────────────
    "H1_SL1.0_TP2+4":      {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "H2_SL1.0_TP1.5+3":    {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 50},
                                           {"pct": 0.03, "qty_pct": 100}]},
    "H3_SL0.8_TP1.5+3":    {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "sl_long_pct": 0.008,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 50},
                                           {"pct": 0.03, "qty_pct": 100}]},

    # ─── I. TP 비율 조정 (1단계 비중 ↑ → 빠른 BEP 확보) ──────────────
    "I1_TP2(70)+4(30)":    {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "I2_TP1.5(70)+3":      {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 70},
                                           {"pct": 0.03, "qty_pct": 100}]},
    "I3_TP2.5+5":          {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ct_long_tp": [{"pct": 0.025, "qty_pct": 50},
                                           {"pct": 0.05, "qty_pct": 100}]},

    # ─── J. 진입 추가 강화 (RSI/연속봉) ─────────────────────────────
    "J1_RSI22_TP2+4":      {"ct_bottom_quality_min": 3, "rsi_long_entry1": 22,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "J2_CONSEC5_TP2+4":    {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── K. EMA 필터 강화 (price ≤ EMA20 — 더 강한 과매도 요구) ────
    "K1_EMA20_TP2+4":      {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ema_filter": {"enabled": True, "length": 20},
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "K2_EMA50_TP2+4":      {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                            "ema_filter": {"enabled": True, "length": 50},
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── L. 복합 최강 (SL 타이트 + 진입 강화 + 1단계 비중 ↑) ─────────
    "L1_SL1_RSI22_TP2(70)": {"ct_bottom_quality_min": 3, "rsi_long_entry1": 22,
                             "sl_long_pct": 0.01,
                             "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                            {"pct": 0.04, "qty_pct": 100}]},
    "L2_SL1_CONSEC5_TP1.5+3": {"ct_bottom_quality_min": 3, "rsi_long_entry1": 25,
                               "consec_candles": 5, "sl_long_pct": 0.01,
                               "ct_long_tp": [{"pct": 0.015, "qty_pct": 50},
                                              {"pct": 0.03, "qty_pct": 100}]},
    "L3_SL1_RSI22_CONSEC5":   {"ct_bottom_quality_min": 3, "rsi_long_entry1": 22,
                               "consec_candles": 5, "sl_long_pct": 0.01,
                               "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                              {"pct": 0.04, "qty_pct": 100}]},
}

# ── 라운드 3 정식 비교 변형 ─────────────────────────────────────────────
LONG_VARIANTS = {
    # 라운드 2 우승들 비교 기준선
    "R2_I1_BASE":          {**_R2_BEST,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "R2_J2_BASE":          {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.04, "qty_pct": 100}]},
    "R2_L1_BASE":          {**_R2_BEST, "rsi_long_entry1": 22, "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M. I1 + J2 (1단계 비중 70 + consec 5) ──────────────────────
    "M1_I1+J2":            {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M2. I1 + SL1.0 (R/R 개선 + 빠른 1단계) ───────────────────
    "M2_I1+SL1":           {**_R2_BEST, "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M3. I1 + J2 + SL1 (3종 결합) ────────────────────────────────
    "M3_I1+J2+SL1":        {**_R2_BEST, "consec_candles": 5, "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 70},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M4. 1단계 비중 80 (더 빠른 BEP) ────────────────────────────
    "M4_TP2(80)+4":        {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 80},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M5. 3단계 TP (2/3/5) - 점진 익절 ──────────────────────────
    "M5_TP2+3+5":          {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.02, "qty_pct": 50},
                                           {"pct": 0.03, "qty_pct": 80},
                                           {"pct": 0.05, "qty_pct": 100}]},

    # ─── M6. 1단계 TP를 더 가깝게 (1.5%) - 거의 모두 익절 ──────────
    "M6_TP1.5(70)+3":      {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 70},
                                           {"pct": 0.03, "qty_pct": 100}]},

    # ─── M7. SL1 + 1.5% 1단계 비중 ↑ + RSI22 ────────────────────────
    "M7_PRECISE_SL1":      {**_R2_BEST, "rsi_long_entry1": 22, "consec_candles": 5,
                            "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 70},
                                           {"pct": 0.03, "qty_pct": 100}]},

    # ─── M8. 비대칭: 1단계 TP 매우 가깝게(1%) + 잔여 4% 추격 ────────
    "M8_TP1(80)+4":        {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.01, "qty_pct": 80},
                                           {"pct": 0.04, "qty_pct": 100}]},

    # ─── M9. 비대칭: 1단계 매우 가까움 + SL 타이트 ──────────────────
    "M9_TP1(70)+3_SL1":    {**_R2_BEST, "consec_candles": 5, "sl_long_pct": 0.01,
                            "ct_long_tp": [{"pct": 0.01, "qty_pct": 70},
                                           {"pct": 0.03, "qty_pct": 100}]},

    # ─── M10. 단일 매우 가까운 TP (1.5% 100% 청산) ─────────────────
    "M10_TP1.5_solo":      {**_R2_BEST, "consec_candles": 5,
                            "ct_long_tp": [{"pct": 0.015, "qty_pct": 100}]},
}


def _ct_metrics(ct_trades: list[dict]) -> dict:
    if not ct_trades:
        return {"n": 0, "wr": 0, "net": 0, "pf": 0,
                "avg_w": 0, "avg_l": 0, "tp": 0, "sl": 0}
    wins   = [t for t in ct_trades if t["net_pnl"] > 0]
    losses = [t for t in ct_trades if t["net_pnl"] <= 0]
    gp     = sum(t["net_pnl"] for t in wins)
    gl     = abs(sum(t["net_pnl"] for t in losses))
    reasons = [t.get("close_reason", "") for t in ct_trades]
    return {
        "n":     len(ct_trades),
        "wr":    len(wins) / len(ct_trades) * 100,
        "net":   sum(t["net_pnl"] for t in ct_trades),
        "pf":    gp / gl if gl > 0 else float("inf"),
        "avg_w": sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0,
        "avg_l": sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0,
        "tp":    reasons.count("TP"),
        "sl":    reasons.count("SL"),
    }


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*72}")
    print(f"  CT LONG 단독 최적화  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  {base_params['start_date']} ~ {base_params['end_date']}")
    print(f"  BTC 필터: 4h EMA50 (btc_bull)  |  변형 {len(LONG_VARIANTS)}종")
    print(f"{'='*72}\n")

    from data.fetcher import load_ohlcv_with_warmup
    from strategy.signal import build_signals
    from indicators.counter_signals import add_counter_signals
    from backtest.engine import BacktestEngine

    print("[1] PI/USDT 데이터 로드...")
    df_raw = load_ohlcv_with_warmup(
        symbol        = base_params["symbol"],
        timeframe     = base_params["timeframe"],
        start_date    = base_params["start_date"],
        end_date      = base_params["end_date"],
        exchange_id   = base_params.get("exchange", "okx"),
        warmup_bars   = 200,
    )

    btc_cfg = base_params.get("btc_filter", {})
    df_btc  = None
    if btc_cfg.get("enabled", False):
        print("[2] BTC 4h 데이터 로드...")
        df_btc = load_ohlcv_with_warmup(
            symbol      = btc_cfg.get("symbol", "BTC/USDT:USDT"),
            timeframe   = btc_cfg.get("timeframe", "4h"),
            start_date  = base_params["start_date"],
            end_date    = base_params["end_date"],
            exchange_id = base_params.get("exchange", "okx"),
            warmup_bars = 200,
        )

    print("[3] 메인 신호 계산 (CT 비활성 1회 공유)...")
    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_with_signals = build_signals(df_raw.copy(), params_no_ct, btc_df=df_btc)
    df_bt_base = df_with_signals[df_with_signals["in_backtest"]].copy()

    if "btc_bull" in df_bt_base.columns:
        bull_pct = df_bt_base["btc_bull"].mean() * 100
        print(f"  btc_bull 비율: {bull_pct:.1f}% (CT LONG 허용 구간)")

    print(f"\n[4] {len(LONG_VARIANTS)}종 변형 백테스트...\n")
    rows = []
    initial = base_params["initial_capital"]

    for name, overrides in LONG_VARIANTS.items():
        ct_cfg = {**BASE_CT, **overrides}

        print(f"  {name:25s} ...", end="", flush=True)

        df_bt = add_counter_signals(df_bt_base.copy(), ct_cfg)

        params = copy.deepcopy(base_params)
        params["counter_trend"] = ct_cfg

        engine = BacktestEngine(params)
        _, ct_trades, eq_curve = engine.run(df_bt)

        m         = _ct_metrics(ct_trades)
        final_eq  = eq_curve[-1]["equity"] if eq_curve else initial
        total_pct = (final_eq / initial - 1) * 100
        rr   = abs(m["avg_w"] / m["avg_l"]) if m["avg_l"] != 0 else 0
        be_wr = 1 / (1 + rr) * 100 if rr > 0 else 0

        marker = "[+]" if m["net"] > 0 else "[-]"
        print(f" {marker} {m['n']:3d}건  WR={m['wr']:5.1f}%  PF={m['pf']:.3f}  "
              f"CT={m['net']:+.0f}$  합산={total_pct:+.0f}%")

        rows.append({
            "설정":        name,
            "건수":        m["n"],
            "WR(%)":       f"{m['wr']:.1f}",
            "PF":          f"{m['pf']:.3f}" if m["pf"] != float("inf") else "∞",
            "R/R":         f"{rr:.2f}",
            "BE_WR":       f"{be_wr:.0f}%",
            "CT순익($)":   f"{m['net']:+.0f}",
            "avg_win":     f"{m['avg_w']:+.1f}",
            "avg_loss":    f"{m['avg_l']:+.1f}",
            "TP/SL":       f"{m['tp']}/{m['sl']}",
            "합산수익(%)": f"{total_pct:+.0f}",
        })

    print(f"\n{'='*72}")
    print("  CT LONG 변형 결과 (CT SHORT는 비활성 상태)")
    print(f"{'='*72}")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    # 흑자 변형만 추출
    profitable = [r for r in rows if float(r["CT순익($)"].replace("+", "")) > 0]
    if profitable:
        print(f"\n  [흑자 변형 {len(profitable)}/{len(rows)}]")
        best_net = max(profitable, key=lambda r: float(r["CT순익($)"].replace("+", "")))
        best_pf  = max(profitable, key=lambda r: float(r["PF"].replace("∞", "9999")))
        print(f"  >> CT순익 최대: [{best_net['설정']}] -> {best_net['CT순익($)']}$ "
              f"/ WR={best_net['WR(%)']}% / PF={best_net['PF']}")
        print(f"  >> PF 최대:    [{best_pf['설정']}] -> PF={best_pf['PF']} "
              f"/ WR={best_pf['WR(%)']}% / {best_pf['CT순익($)']}$")
    else:
        print("\n  [경고] 모든 변형이 적자 - 베어장에서 CT LONG 자체가 부적합")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
