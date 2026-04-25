"""
CT 최적화 비교 (BTC 필터 적용 상태에서 변형 테스트)

현재 기준선: consec=4, RSI65, SL=1.5%, TP=10%, BTC EMA50 필터
→ 95건, WR=23.16%, PF=1.865, CT +$1,010

테스트 방향:
  A. TP 낮추기        (6~8%) → 빠른 TP 도달, 포지션 회전 증가
  B. DCA 1회 허용     → 평균가 낮춰 TP 도달 확률 상승
  C. 2단계 TP         → 일부 빠른 수익 + 잔여 보유
  D. 진입 완화        → RSI60 / consec3 / quality_min1

사용법:
  python ct_optimize.py
"""

import copy
import json

from tabulate import tabulate


# ── 기준선 CT 파라미터 (현재 최적) ──────────────────────────────────────────
BASE_CT = {
    "enabled":               True,
    "equity_pct":            0,          # 레버리지 모드
    "rsi_period":            14,
    "consec_candles":        4,
    "min_candle_pct":        0.0,
    "ema_filter":            {"enabled": True, "length": 30},
    "divergence_pivot_period": 5,
    "divergence_max_bars":   60,
    "ct_long_enabled":       False,
    "ct_short_enabled":      True,
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
    "ct_long_tp":  [{"pct": 0.10, "qty_pct": 100}],
    "ct_short_tp": [{"pct": 0.10, "qty_pct": 100}],
}

# ── 변형 정의 (기준선 위에 덮어쓸 값만 지정) ────────────────────────────────
VARIANTS = {
    "BASE_현재":       {},

    # A. TP 낮추기 (거래수·WR 동시 상승 기대)
    "A1_TP8%":         {"ct_short_tp": [{"pct": 0.08, "qty_pct": 100}]},
    "A2_TP7%":         {"ct_short_tp": [{"pct": 0.07, "qty_pct": 100}]},
    "A3_TP6%":         {"ct_short_tp": [{"pct": 0.06, "qty_pct": 100}]},

    # B. DCA 1회 (평균가 낮춰 TP 도달 확률 상승)
    "B1_DCA_1.5%":     {"max_dca": 1, "dca_price_pct": 0.015, "dca_weights": [1]},
    "B2_DCA_2.0%":     {"max_dca": 1, "dca_price_pct": 0.020, "dca_weights": [1]},

    # C. 2단계 TP (일부 빠른 수익 실현 + 잔여 보유)
    "C1_2TP_5+10":     {"ct_short_tp": [{"pct": 0.05, "qty_pct": 50},
                                         {"pct": 0.10, "qty_pct": 100}]},
    "C2_2TP_7+12":     {"ct_short_tp": [{"pct": 0.07, "qty_pct": 50},
                                         {"pct": 0.12, "qty_pct": 100}]},

    # D. 진입 완화 (거래수 증가 탐색)
    "D1_RSI60":        {"rsi_short_entry1": 60},
    "D2_CONSEC3":      {"consec_candles": 3},
    "D3_QUAL1":        {"ct_bottom_quality_min": 1},

    # E. 복합 최적 조합 후보 (A+D)
    "E1_TP8_RSI60":    {"ct_short_tp": [{"pct": 0.08, "qty_pct": 100}],
                        "rsi_short_entry1": 60},
    "E2_TP7_CONSEC3":  {"ct_short_tp": [{"pct": 0.07, "qty_pct": 100}],
                        "consec_candles": 3},
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
    print(f"  CT 최적화 비교  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  {base_params['start_date']} ~ {base_params['end_date']}")
    print(f"  BTC 필터: 4h EMA50  |  변형 {len(VARIANTS)}종")
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

    print("[3] 메인 신호 계산 (CT 비활성화 상태, 1회 공유)...")
    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_with_signals = build_signals(df_raw.copy(), params_no_ct, btc_df=df_btc)
    df_bt_base = df_with_signals[df_with_signals["in_backtest"]].copy()
    print(f"  btc_bear 비율: {df_bt_base.get('btc_bear', 0).mean()*100:.1f}%"
          if "btc_bear" in df_bt_base.columns else "  BTC 필터 없음")

    print(f"\n[4] {len(VARIANTS)}종 변형 백테스트...\n")
    rows = []
    initial = base_params["initial_capital"]

    for name, overrides in VARIANTS.items():
        ct_cfg = {**BASE_CT, **overrides}
        # 중첩 dict는 별도 merge
        if "ema_filter" in overrides:
            ct_cfg["ema_filter"] = {**BASE_CT["ema_filter"], **overrides["ema_filter"]}

        print(f"  {name:20s} ...", end="", flush=True)

        df_bt = add_counter_signals(df_bt_base.copy(), ct_cfg)

        params = copy.deepcopy(base_params)
        params["counter_trend"] = ct_cfg

        engine = BacktestEngine(params)
        _, ct_trades, eq_curve = engine.run(df_bt)

        m       = _ct_metrics(ct_trades)
        final_eq = eq_curve[-1]["equity"] if eq_curve else initial
        total_pct = (final_eq / initial - 1) * 100
        rr = abs(m["avg_w"] / m["avg_l"]) if m["avg_l"] != 0 else 0
        be_wr = 1 / (1 + rr) * 100 if rr > 0 else 0

        print(f" {m['n']:3d}건  WR={m['wr']:5.1f}%  PF={m['pf']:.3f}  "
              f"CT={m['net']:+.0f}$  합산={total_pct:+.0f}%")

        rows.append({
            "설정":        name,
            "건수":        m["n"],
            "WR(%)":       f"{m['wr']:.1f}",
            "PF":          f"{m['pf']:.3f}" if m["pf"] != float("inf") else "∞",
            "R/R":         f"{rr:.1f}",
            "손익분기WR":  f"{be_wr:.0f}%",
            "CT순익($)":   f"{m['net']:+.0f}",
            "avg_win":     f"{m['avg_w']:+.1f}",
            "avg_loss":    f"{m['avg_l']:+.1f}",
            "TP/SL":       f"{m['tp']}/{m['sl']}",
            "합산수익(%)": f"{total_pct:+.0f}",
        })

    print(f"\n{'='*72}")
    print("  결과 비교표")
    print(f"{'='*72}")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    # 최우수 추천
    best_net = max(rows, key=lambda r: float(r["CT순익($)"].replace("+", "")))
    best_pf  = max(rows, key=lambda r: float(r["PF"].replace("∞", "9999")))
    best_tot = max(rows, key=lambda r: float(r["합산수익(%)"].replace("+", "")))

    print(f"\n  ★ CT순익 최대: [{best_net['설정']}]  "
          f"→ {best_net['CT순익($)']}$ / WR={best_net['WR(%)']}% / PF={best_net['PF']}")
    print(f"  ★ PF 최대:    [{best_pf['설정']}]  "
          f"→ PF={best_pf['PF']} / WR={best_pf['WR(%)']}% / {best_pf['CT순익($)']}$")
    print(f"  ★ 합산수익 최대: [{best_tot['설정']}]  "
          f"→ {best_tot['합산수익(%)']}% / CT={best_tot['CT순익($)']}$")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
