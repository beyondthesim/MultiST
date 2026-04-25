"""
C2_FIB_DCA10 사이즈 스케일링 비교

기준 변형: max_dca=10, dca_weights=피보나치, SL=10%, TP=3%
ct_size_pct를 100/200/500/1000/2000% 스윕하여 사이즈가 결과에 미치는 영향 측정.

[주의] C2_FIB는 SL이 1번만 발생해 PF=37 비현실적 수치 → 통계 신뢰성 낮음.
사이즈 키울수록 SL 1회당 손실도 비례 증가하므로 변동성 위험 함께 평가.
"""

import copy
import json

from tabulate import tabulate


def _fib(n: int) -> list[int]:
    out = [1, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


# C2_FIB_DCA10 고정 + ct_size_pct만 변경
C2_FIB_BASE = {
    "enabled":               True,
    "equity_pct":            0,
    "rsi_period":            14,
    "consec_candles":        4,
    "consec_candles_long":   5,
    "consec_candles_short":  4,
    "min_candle_pct":        0.0,
    "ema_filter":            {"enabled": True, "length": 30},
    "divergence_pivot_period": 5,
    "divergence_max_bars":   60,
    "ct_long_enabled":       True,
    "ct_short_enabled":      True,
    "rsi_long_entry1":       25,
    "rsi_short_entry1":      65,
    "rsi_long_exit":         65,
    "rsi_short_exit":        35,
    "ct_exit_enabled":       False,
    # ── C2_FIB 핵심 ─────────────────────────────────────
    "max_dca":               10,
    "dca_weights":           _fib(10),  # [1,1,2,3,5,8,13,21,34,55] sum=143
    "dca_price_pct":         0.015,
    "dca_require_divergence": False,
    "sl_long_pct":           0.10,
    "sl_short_pct":          0.10,
    "ct_long_tp":  [{"pct": 0.03, "qty_pct": 100}],
    "ct_short_tp": [{"pct": 0.03, "qty_pct": 100}],
    # ────────────────────────────────────────────────────
    "all_close_pct":         0.10,
    "safe_close_count":      2,
    "safe_close_pct":        0.02,
    "ct_bottom_quality_min":       2,
    "ct_bottom_quality_min_long":  3,
    "ct_bottom_quality_min_short": 2,
}


SIZE_VARIANTS = {
    "C2_FIB_x1":     {"ct_size_pct": 100},      # 현재 기준 (ct_base $1,000)
    "C2_FIB_x2":     {"ct_size_pct": 200},      # ct_base $2,000
    "C2_FIB_x5":     {"ct_size_pct": 500},      # ct_base $5,000
    "C2_FIB_x10":    {"ct_size_pct": 1000},     # ct_base $10,000  ★ 사용자 요청
    "C2_FIB_x15":    {"ct_size_pct": 1500},     # ct_base $15,000
    "C2_FIB_x20":    {"ct_size_pct": 2000},     # ct_base $20,000
}


def _ct_metrics(ct_trades: list[dict]) -> dict:
    if not ct_trades:
        return {k: 0 for k in ["n", "wr", "net", "pf", "avg_w", "avg_l",
                               "max_w", "max_l", "tp", "sl"]}
    wins   = [t for t in ct_trades if t["net_pnl"] > 0]
    losses = [t for t in ct_trades if t["net_pnl"] <= 0]
    gp = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in losses))
    reasons = [t.get("close_reason", "") for t in ct_trades]
    return {
        "n":     len(ct_trades),
        "wr":    len(wins) / len(ct_trades) * 100,
        "net":   sum(t["net_pnl"] for t in ct_trades),
        "pf":    gp / gl if gl > 0 else float("inf"),
        "avg_w": gp / len(wins) if wins else 0,
        "avg_l": -gl / len(losses) if losses else 0,
        "max_w": max((t["net_pnl"] for t in wins),   default=0),
        "max_l": min((t["net_pnl"] for t in losses), default=0),
        "tp":    reasons.count("TP"),
        "sl":    reasons.count("SL"),
    }


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*82}")
    print(f"  C2_FIB_DCA10 사이즈 스케일링  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  설정: max_dca=10, fib weights, SL=10%, TP=3% (평균단가 기준)")
    print(f"  사이즈 스윕: x1 ~ x20  (ct_base = $1,000 ~ $20,000)")
    print(f"{'='*82}\n")

    from data.fetcher import load_ohlcv_with_warmup
    from strategy.signal import build_signals
    from indicators.counter_signals import add_counter_signals
    from backtest.engine import BacktestEngine

    print("[1] 데이터 로드...")
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
        df_btc = load_ohlcv_with_warmup(
            symbol      = btc_cfg.get("symbol", "BTC/USDT:USDT"),
            timeframe   = btc_cfg.get("timeframe", "4h"),
            start_date  = base_params["start_date"],
            end_date    = base_params["end_date"],
            exchange_id = base_params.get("exchange", "okx"),
            warmup_bars = 200,
        )

    print("[2] 메인 신호 (1회 공유)...")
    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_with_signals = build_signals(df_raw.copy(), params_no_ct, btc_df=df_btc)
    df_bt_base = df_with_signals[df_with_signals["in_backtest"]].copy()

    # CT 신호는 사이즈와 무관하므로 1회만 계산
    df_bt = add_counter_signals(df_bt_base.copy(), C2_FIB_BASE)

    print(f"\n[3] {len(SIZE_VARIANTS)}종 사이즈 백테스트...\n")
    rows = []
    initial = base_params["initial_capital"]

    for name, overrides in SIZE_VARIANTS.items():
        ct_cfg = {**C2_FIB_BASE, **overrides}

        params = copy.deepcopy(base_params)
        params["counter_trend"] = ct_cfg

        engine = BacktestEngine(params)
        _, ct_trades, eq_curve = engine.run(df_bt)

        m         = _ct_metrics(ct_trades)
        final_eq  = eq_curve[-1]["equity"] if eq_curve else initial
        total_pct = (final_eq / initial - 1) * 100
        ct_base   = initial * (overrides["ct_size_pct"] / 100.0)

        # Max DD 계산
        eq_vals = [e["equity"] for e in eq_curve]
        if eq_vals:
            peak = eq_vals[0]
            max_dd = 0.0
            for v in eq_vals:
                peak = max(peak, v)
                dd = (peak - v) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0.0

        marker = "[+]" if m["net"] > 0 else "[-]"
        print(f"  {name:14s} ct_base=${ct_base:>7,.0f}  {marker} "
              f"{m['n']:3d}건  WR={m['wr']:5.1f}%  PF={m['pf']:7.3f}  "
              f"CT={m['net']:+7.0f}$  최대손실={m['max_l']:+7.1f}  DD={max_dd:5.2f}%  "
              f"합산={total_pct:+5.0f}%")

        rows.append({
            "설정":         name,
            "ct_base($)":   f"{ct_base:,.0f}",
            "건수":         m["n"],
            "WR(%)":        f"{m['wr']:.1f}",
            "PF":           f"{m['pf']:.3f}" if m["pf"] != float("inf") else "∞",
            "CT순익($)":    f"{m['net']:+.0f}",
            "avg_win":      f"{m['avg_w']:+.1f}",
            "avg_loss":     f"{m['avg_l']:+.1f}",
            "최대수익":     f"{m['max_w']:+.1f}",
            "최대손실":     f"{m['max_l']:+.1f}",
            "TP/SL":        f"{m['tp']}/{m['sl']}",
            "최대DD(%)":    f"{max_dd:.2f}",
            "합산수익(%)":  f"{total_pct:+.0f}",
        })

    print(f"\n{'='*82}")
    print("  C2_FIB 사이즈 스윕 결과")
    print(f"{'='*82}")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    # BASE_DCA0 (현재 운용 기준선)과의 비교
    print(f"\n  [참고] BASE_DCA0 (현재 운용): CT +$1,020 / WR 36.0% / PF 1.873 / 합산 +1,972%")
    print(f"{'='*82}\n")


if __name__ == "__main__":
    main()
