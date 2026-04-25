"""
CT 물타기(DCA) 최적화 비교

DCA 차원:
  - max_dca: 0/3/5/10/15/20  (물타기 횟수)
  - dca_weights: 균등 / 점진증가 / 피보나치
  - dca_price_pct: 1.5% (기본)
  - 양방향 적용 (CT LONG + CT SHORT)

엔진 동작:
  - DCA 시 평균단가 재계산 → SL/TP도 평균단가 기준 재설정
  - 각 DCA 진입 사이즈 = ct_per_entry × dca_weights[k]
  - 가격 트리거: 직전 매수가 대비 dca_price_pct% 추가 역행
  - 다이버전스 요구: dca_require_divergence (False 권장)

사용법: python dca_optimize.py
"""

import copy
import json

from tabulate import tabulate


# 기준 CT 설정 (현재 params.json과 동일, DCA만 변경)
BASE_CT = {
    "enabled":               True,
    "equity_pct":            0,
    "ct_size_pct":           100,
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
    "max_dca":               0,
    "dca_weights":           [],
    "dca_price_pct":         0.015,
    "dca_require_divergence": False,
    "sl_long_pct":           0.015,
    "sl_short_pct":          0.015,
    "all_close_pct":         0.10,
    "safe_close_count":      2,
    "safe_close_pct":        0.02,
    "ct_bottom_quality_min":       2,
    "ct_bottom_quality_min_long":  3,
    "ct_bottom_quality_min_short": 2,
    "ct_long_tp":  [{"pct": 0.02, "qty_pct": 80}, {"pct": 0.04, "qty_pct": 100}],
    "ct_short_tp": [{"pct": 0.07, "qty_pct": 50}, {"pct": 0.12, "qty_pct": 100}],
}


def _eq(n: int) -> list[int]:
    """균등 가중치 [1]*n"""
    return [1] * n


def _grad(n: int) -> list[int]:
    """점진 증가 [1,1,2,2,3,3,...]"""
    return [(i // 2) + 1 for i in range(n)]


def _fib(n: int) -> list[int]:
    """피보나치 1,1,2,3,5,8,13,21,..."""
    out = [1, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


# ── DCA 친화 설계 변형 (라운드 2) ─────────────────────────────────────
# SL을 넓히고 TP를 작게 (평균단가 기준) → DCA 활용 + 작은 반등으로 익절
# 기준 BASE는 SL=1.5% TP=비대칭 (DCA 안 발동되는 설계)

def _ct_tp(pct: float) -> list[dict]:
    return [{"pct": pct, "qty_pct": 100}]


DCA_VARIANTS = {
    "BASE_DCA0":              {"max_dca": 0, "dca_weights": []},

    # ─── A. SL 넓힘 + 평균가 기준 작은 TP + DCA 활성 ──────────────
    "A1_SL3_TP2_DCA10":       {"max_dca": 10, "dca_weights": _eq(10),
                                "sl_long_pct": 0.03, "sl_short_pct": 0.03,
                                "ct_long_tp":  _ct_tp(0.02),
                                "ct_short_tp": _ct_tp(0.02)},

    "A2_SL5_TP2_DCA10":       {"max_dca": 10, "dca_weights": _eq(10),
                                "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                "ct_long_tp":  _ct_tp(0.02),
                                "ct_short_tp": _ct_tp(0.02)},

    "A3_SL5_TP3_DCA10":       {"max_dca": 10, "dca_weights": _eq(10),
                                "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                "ct_long_tp":  _ct_tp(0.03),
                                "ct_short_tp": _ct_tp(0.03)},

    "A4_SL5_TP2_DCA20_PX1":   {"max_dca": 20, "dca_weights": _eq(20),
                                "dca_price_pct": 0.01,
                                "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                "ct_long_tp":  _ct_tp(0.02),
                                "ct_short_tp": _ct_tp(0.02)},

    "A5_SL10_TP3_DCA20":      {"max_dca": 20, "dca_weights": _eq(20),
                                "sl_long_pct": 0.10, "sl_short_pct": 0.10,
                                "ct_long_tp":  _ct_tp(0.03),
                                "ct_short_tp": _ct_tp(0.03)},

    # ─── B. 공격적 DCA (SL 매우 넓음) ────────────────────────────────
    "B1_SL10_TP2_DCA20_PX1":  {"max_dca": 20, "dca_weights": _eq(20),
                                "dca_price_pct": 0.01,
                                "sl_long_pct": 0.10, "sl_short_pct": 0.10,
                                "ct_long_tp":  _ct_tp(0.02),
                                "ct_short_tp": _ct_tp(0.02)},

    "B2_SL15_TP3_DCA20_PX1":  {"max_dca": 20, "dca_weights": _eq(20),
                                "dca_price_pct": 0.01,
                                "sl_long_pct": 0.15, "sl_short_pct": 0.15,
                                "ct_long_tp":  _ct_tp(0.03),
                                "ct_short_tp": _ct_tp(0.03)},

    # ─── C. 점진/피보나치 가중치 (뒤로 갈수록 큰 진입) ──────────────
    "C1_GRAD_SL5_TP2_DCA10":  {"max_dca": 10, "dca_weights": _grad(10),
                                "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                "ct_long_tp":  _ct_tp(0.02),
                                "ct_short_tp": _ct_tp(0.02)},

    "C2_FIB_SL10_TP3_DCA10":  {"max_dca": 10, "dca_weights": _fib(10),
                                "sl_long_pct": 0.10, "sl_short_pct": 0.10,
                                "ct_long_tp":  _ct_tp(0.03),
                                "ct_short_tp": _ct_tp(0.03)},

    # ─── D. 그리드 트레이딩 (매우 작은 TP, 자주 DCA) ────────────────
    "D1_GRID_TP1.5_SL5_DCA10": {"max_dca": 10, "dca_weights": _eq(10),
                                 "dca_price_pct": 0.015,
                                 "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                 "ct_long_tp":  _ct_tp(0.015),
                                 "ct_short_tp": _ct_tp(0.015)},

    "D2_GRID_TP1_SL5_DCA15_PX1": {"max_dca": 15, "dca_weights": _eq(15),
                                   "dca_price_pct": 0.01,
                                   "sl_long_pct": 0.05, "sl_short_pct": 0.05,
                                   "ct_long_tp":  _ct_tp(0.01),
                                   "ct_short_tp": _ct_tp(0.01)},

    "D3_GRID_TP2_SL8_DCA20_PX1.5": {"max_dca": 20, "dca_weights": _eq(20),
                                     "dca_price_pct": 0.015,
                                     "sl_long_pct": 0.08, "sl_short_pct": 0.08,
                                     "ct_long_tp":  _ct_tp(0.02),
                                     "ct_short_tp": _ct_tp(0.02)},
}


def _ct_metrics(ct_trades: list[dict]) -> dict:
    if not ct_trades:
        return {k: 0 for k in ["n", "wr", "net", "pf", "avg_w", "avg_l",
                               "tp", "sl", "dca_used", "max_dca_seen"]}
    wins   = [t for t in ct_trades if t["net_pnl"] > 0]
    losses = [t for t in ct_trades if t["net_pnl"] <= 0]
    gp = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in losses))
    reasons = [t.get("close_reason", "") for t in ct_trades]
    dca_counts = [t.get("dca_count", 0) for t in ct_trades]
    return {
        "n":            len(ct_trades),
        "wr":           len(wins) / len(ct_trades) * 100,
        "net":          sum(t["net_pnl"] for t in ct_trades),
        "pf":           gp / gl if gl > 0 else float("inf"),
        "avg_w":        sum(t["net_pnl"] for t in wins)   / len(wins)   if wins   else 0,
        "avg_l":        sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0,
        "tp":           reasons.count("TP"),
        "sl":           reasons.count("SL"),
        "dca_used":     sum(1 for c in dca_counts if c > 0),
        "max_dca_seen": max(dca_counts) if dca_counts else 0,
    }


def main():
    with open("config/params.json", encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*80}")
    print(f"  CT 물타기(DCA) 최적화  |  {base_params['symbol']}  {base_params['timeframe']}")
    print(f"  {base_params['start_date']} ~ {base_params['end_date']}")
    print(f"  변형 {len(DCA_VARIANTS)}종  (양방향 CT 활성, ct_size_pct=100)")
    print(f"{'='*80}\n")

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

    print("[2] 메인 신호 계산 (1회 공유)...")
    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_with_signals = build_signals(df_raw.copy(), params_no_ct, btc_df=df_btc)
    df_bt_base = df_with_signals[df_with_signals["in_backtest"]].copy()

    print(f"\n[3] {len(DCA_VARIANTS)}종 DCA 변형 백테스트...\n")
    rows = []
    initial = base_params["initial_capital"]

    for name, overrides in DCA_VARIANTS.items():
        ct_cfg = {**BASE_CT, **overrides}

        print(f"  {name:20s} ...", end="", flush=True)

        df_bt = add_counter_signals(df_bt_base.copy(), ct_cfg)

        params = copy.deepcopy(base_params)
        params["counter_trend"] = ct_cfg

        engine = BacktestEngine(params)
        _, ct_trades, eq_curve = engine.run(df_bt)

        m         = _ct_metrics(ct_trades)
        final_eq  = eq_curve[-1]["equity"] if eq_curve else initial
        total_pct = (final_eq / initial - 1) * 100

        marker = "[+]" if m["net"] > 0 else "[-]"
        print(f" {marker} {m['n']:3d}건  WR={m['wr']:5.1f}%  PF={m['pf']:.3f}  "
              f"CT={m['net']:+6.0f}$  DCA사용={m['dca_used']:3d}건/최대{m['max_dca_seen']}회  "
              f"합산={total_pct:+.0f}%")

        rows.append({
            "설정":         name,
            "건수":         m["n"],
            "WR(%)":        f"{m['wr']:.1f}",
            "PF":           f"{m['pf']:.3f}" if m["pf"] != float("inf") else "∞",
            "CT순익($)":    f"{m['net']:+.0f}",
            "avg_win":      f"{m['avg_w']:+.1f}",
            "avg_loss":     f"{m['avg_l']:+.1f}",
            "TP/SL":        f"{m['tp']}/{m['sl']}",
            "DCA건/최대":   f"{m['dca_used']}/{m['max_dca_seen']}",
            "합산수익(%)":  f"{total_pct:+.0f}",
        })

    print(f"\n{'='*80}")
    print("  DCA 변형 결과")
    print(f"{'='*80}")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    profitable = [r for r in rows if float(r["CT순익($)"].replace("+", "")) > 0]
    if profitable:
        best = max(profitable, key=lambda r: float(r["합산수익(%)"].replace("+", "")))
        print(f"\n  >> 합산수익 최대: [{best['설정']}] -> {best['합산수익(%)']}% "
              f"/ CT={best['CT순익($)']}$ / WR={best['WR(%)']}% / PF={best['PF']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
