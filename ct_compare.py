"""
역추세(CT) 전략 파라미터 비교 분석

11가지 CT 설정을 백테스트하여 폭발적 수익 구조 탐색.
메인 전략(factor=5.4, ATR=10)은 고정, CT 파라미터만 변경.

[진단] A_기본의 근본 문제:
  avg_win=1.56 / avg_loss=3.96 → 손익비 0.39
  손익분기 승률 = 3.96/(3.96+1.56) = 71.7% → 실제 66%로 불가
  DCA 2회 → SL 맞으면 3배 포지션이 한번에 손실

[개선 방향] G~K:
  G_균형형     : TP=SL 수준(3%/5%), DCA 최소화, 손익비 1.5 목표
  H_노DCA완전  : DCA 없음 + 타이트SL(1.5%) + 와이드TP(2.5%/5%)
  I_RSI주도청산: RSI 55 청산, TP 12%는 안전망, SL 4%, DCA 1회(극단)
  J_초선별무DCA: 5연속봉+RSI22, DCA 없음, TP 4%/8%, SL 2.5%
  K_하이브리드 : RSI25 진입, 극단DCA 1회(RSI15), SL 2%, TP 3%/6%

사용법:
  python ct_compare.py
  python ct_compare.py --config config/params.json
"""

import argparse
import copy
import json

from tabulate import tabulate

# ── CT 비교 대상 6가지 변형 정의 ─────────────────────────────────────────
CT_VARIANTS = {
    "A_기본": {
        # 현재 기본값 - 비교 기준선
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 30, "rsi_short_entry1": 70,
        "rsi_long_exit": 65,   "rsi_short_exit": 35,
        "max_dca": 2, "rsi_long_dca": [22, 15], "rsi_short_dca": [78, 85],
        "sl_long_pct": 0.03, "sl_short_pct": 0.03,
        "ct_long_tp":  [{"pct": 0.015, "qty_pct": 50}, {"pct": 0.025, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.015, "qty_pct": 50}, {"pct": 0.025, "qty_pct": 100}],
    },
    "B_TP확대": {
        # TP를 SL과 비슷한 수준으로 확대 → 손익비 개선
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 30, "rsi_short_entry1": 70,
        "rsi_long_exit": 65,   "rsi_short_exit": 35,
        "max_dca": 2, "rsi_long_dca": [22, 15], "rsi_short_dca": [78, 85],
        "sl_long_pct": 0.03, "sl_short_pct": 0.03,
        "ct_long_tp":  [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
    },
    "C_RSI출구": {
        # RSI 회복 시 출구 - 고정 TP 없이 추세 반전까지 보유 (폭발적 수익 목표)
        # 매우 넓은 TP(10%)로 실질적으로 RSI 출구가 먼저 작동
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 25, "rsi_short_entry1": 75,
        "rsi_long_exit": 58,   "rsi_short_exit": 42,
        "max_dca": 2, "rsi_long_dca": [18, 12], "rsi_short_dca": [82, 88],
        "sl_long_pct": 0.04, "sl_short_pct": 0.04,
        "ct_long_tp":  [{"pct": 0.10, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.10, "qty_pct": 100}],
    },
    "D_극단RSI": {
        # 진입 조건을 극도로 엄격하게 - RSI ≤ 20만 진입하여 최고 바닥만 포착
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 20, "rsi_short_entry1": 80,
        "rsi_long_exit": 60,   "rsi_short_exit": 40,
        "max_dca": 1, "rsi_long_dca": [12], "rsi_short_dca": [88],
        "sl_long_pct": 0.03, "sl_short_pct": 0.03,
        "ct_long_tp":  [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
    },
    "E_타이트SL": {
        # SL을 TP와 비슷하게 줄여 손익비 1:1 이상 보장 (DCA 없음)
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 30, "rsi_short_entry1": 70,
        "rsi_long_exit": 65,   "rsi_short_exit": 35,
        "max_dca": 0, "rsi_long_dca": [], "rsi_short_dca": [],
        "sl_long_pct": 0.015, "sl_short_pct": 0.015,
        "ct_long_tp":  [{"pct": 0.02, "qty_pct": 50}, {"pct": 0.03, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.02, "qty_pct": 50}, {"pct": 0.03, "qty_pct": 100}],
    },
    "F_연속봉3": {
        # 진입 기준 완화 (3연속 음봉) - 더 많은 기회 포착, DCA 1회
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 3,
        "rsi_long_entry1": 32, "rsi_short_entry1": 68,
        "rsi_long_exit": 62,   "rsi_short_exit": 38,
        "max_dca": 1, "rsi_long_dca": [22], "rsi_short_dca": [78],
        "sl_long_pct": 0.025, "sl_short_pct": 0.025,
        "ct_long_tp":  [{"pct": 0.02, "qty_pct": 50}, {"pct": 0.035, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.02, "qty_pct": 50}, {"pct": 0.035, "qty_pct": 100}],
    },

    # ── 개선 시리즈 (G~K) ───────────────────────────────────────────────────
    # 핵심: 손익비 수정 → avg_win / |avg_loss| > (1-WR)/WR 조건 충족
    # A_기본 손익분기 요구 승률: 3.96/(3.96+1.56)=71.7% → 실제 66% → 적자 구조
    # 목표: SL ≤ TP × (WR/(1-WR)) 또는 TP 대폭 확대로 구조적 흑자 전환

    "G_균형형": {
        # TP를 SL(2%) 이상으로 맞춤, DCA 극단값(RSI18)만 허용
        # 손익비 1.5~2.5 목표 → 승률 40%만 넘어도 수익
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 28, "rsi_short_entry1": 72,
        "rsi_long_exit": 62,   "rsi_short_exit": 38,
        "max_dca": 1, "rsi_long_dca": [18], "rsi_short_dca": [82],
        "sl_long_pct": 0.02, "sl_short_pct": 0.02,
        "ct_long_tp":  [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.03, "qty_pct": 50}, {"pct": 0.05, "qty_pct": 100}],
    },
    "H_노DCA완전": {
        # DCA 완전 제거 → SL 맞아도 단일 포지션만 손실
        # SL 1.5% 타이트 → 손실 크기 절반으로 축소
        # TP 2.5%/5% → 손익비 1.67~3.33
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 28, "rsi_short_entry1": 72,
        "rsi_long_exit": 60,   "rsi_short_exit": 40,
        "max_dca": 0, "rsi_long_dca": [], "rsi_short_dca": [],
        "sl_long_pct": 0.015, "sl_short_pct": 0.015,
        "ct_long_tp":  [{"pct": 0.025, "qty_pct": 40}, {"pct": 0.05, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.025, "qty_pct": 40}, {"pct": 0.05, "qty_pct": 100}],
    },
    "I_RSI주도청산": {
        # RSI 55 회복 시 즉시 청산 → 반등 끝까지 타기
        # TP 12%는 안전망(사실상 RSI 청산이 먼저 작동)
        # SL 4% + DCA 1회(RSI15 극단) → 충분한 여유
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 25, "rsi_short_entry1": 75,
        "rsi_long_exit": 55,   "rsi_short_exit": 45,
        "max_dca": 1, "rsi_long_dca": [15], "rsi_short_dca": [85],
        "sl_long_pct": 0.04, "sl_short_pct": 0.04,
        "ct_long_tp":  [{"pct": 0.12, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.12, "qty_pct": 100}],
    },
    "J_초선별무DCA": {
        # 5연속 음봉 + RSI22 이하 → 최고 품질 신호만 선별
        # DCA 없음 → 리스크 고정, TP 4%/8% → 손익비 1.6~3.2
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 5,
        "rsi_long_entry1": 22, "rsi_short_entry1": 78,
        "rsi_long_exit": 60,   "rsi_short_exit": 40,
        "max_dca": 0, "rsi_long_dca": [], "rsi_short_dca": [],
        "sl_long_pct": 0.025, "sl_short_pct": 0.025,
        "ct_long_tp":  [{"pct": 0.04, "qty_pct": 50}, {"pct": 0.08, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.04, "qty_pct": 50}, {"pct": 0.08, "qty_pct": 100}],
    },
    "K_하이브리드": {
        # 최적 조합: RSI25 진입, RSI15 극단 DCA 1회, SL 2%, TP 3%/6%
        # RSI58 회복 → 빠른 출구로 CT_EXIT 수익 극대화
        # 손익비 1.5~3.0, DCA 단 1회로 리스크 제한
        "equity_pct": 30,
        "rsi_period": 14, "consec_candles": 4,
        "rsi_long_entry1": 25, "rsi_short_entry1": 75,
        "rsi_long_exit": 58,   "rsi_short_exit": 42,
        "max_dca": 1, "rsi_long_dca": [15], "rsi_short_dca": [85],
        "sl_long_pct": 0.02, "sl_short_pct": 0.02,
        "ct_long_tp":  [{"pct": 0.03, "qty_pct": 40}, {"pct": 0.06, "qty_pct": 100}],
        "ct_short_tp": [{"pct": 0.03, "qty_pct": 40}, {"pct": 0.06, "qty_pct": 100}],
    },
}


def _simple_ct_metrics(ct_trades: list[dict], ct_init: float) -> dict:
    if not ct_trades:
        return {k: 0 for k in ["n", "wr", "net_pnl", "net_pct", "pf", "avg_win", "avg_loss",
                                "dca_trades", "tp_exits", "sl_exits", "ct_exits"]}
    winners = [t for t in ct_trades if t["net_pnl"] > 0]
    losers  = [t for t in ct_trades if t["net_pnl"] <= 0]
    gp = sum(t["net_pnl"] for t in winners)
    gl = abs(sum(t["net_pnl"] for t in losers))
    net = sum(t["net_pnl"] for t in ct_trades)
    reasons = [t.get("close_reason", "") for t in ct_trades]
    return {
        "n":          len(ct_trades),
        "wr":         len(winners) / len(ct_trades) * 100,
        "net_pnl":    net,
        "net_pct":    net / ct_init * 100,
        "pf":         gp / gl if gl > 0 else float("inf"),
        "avg_win":    sum(t["net_pnl"] for t in winners) / len(winners) if winners else 0,
        "avg_loss":   sum(t["net_pnl"] for t in losers)  / len(losers)  if losers  else 0,
        "dca_trades": sum(1 for t in ct_trades if t.get("dca_count", 0) > 0),
        "tp_exits":   reasons.count("TP_FULL"),
        "sl_exits":   reasons.count("SL"),
        "ct_exits":   reasons.count("CT_EXIT"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_params = json.load(f)

    print(f"\n{'='*70}")
    print(f"  역추세(CT) 전략 비교 분석")
    print(f"  심볼: {base_params['symbol']}  TF: {base_params['timeframe']}")
    print(f"  기간: {base_params['start_date']} ~ {base_params['end_date']}")
    print(f"  메인 ST: factor={base_params['st1']['factor']}  ATR={base_params['st1']['atr_period']}")
    print(f"  비교 대상: {len(CT_VARIANTS)}종 CT 설정")
    print(f"{'='*70}\n")

    # ── 데이터 & 메인 신호 공유 (한 번만 로드) ───────────────────────────
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
        force_refresh = False,
    )
    print(f"  봉 수: {len(df_raw):,}")

    # 메인 신호 (CT 비활성화 상태로 1회 계산)
    print("[2] 메인 신호 계산 (공유)...")
    params_no_ct = copy.deepcopy(base_params)
    params_no_ct["counter_trend"] = {"enabled": False}
    df_main_signals = build_signals(df_raw.copy(), params_no_ct)
    df_bt_base = df_main_signals[df_main_signals["in_backtest"]].copy()

    ct_init = base_params["initial_capital"] * 0.30  # CT 자본 (30% 기준)

    print(f"\n[3] CT 변형별 백테스트 시작...\n")
    summary_rows = []

    for name, ct_cfg in CT_VARIANTS.items():
        print(f"  └ {name} 실행 중...", end="", flush=True)

        # CT 신호 재계산 (CT 설정별로 다름)
        df_bt = add_counter_signals(df_bt_base.copy(), ct_cfg)

        # CT 활성화된 params 구성
        params = copy.deepcopy(base_params)
        params["counter_trend"] = {"enabled": True, **ct_cfg}

        engine = BacktestEngine(params)
        main_trades, ct_trades, eq_curve = engine.run(df_bt)

        # 합산 성과
        initial   = base_params["initial_capital"]
        final_eq  = eq_curve[-1]["equity"] if eq_curve else initial
        combined_pct = (final_eq / initial - 1) * 100

        # CT 단독 성과
        ct_m = _simple_ct_metrics(ct_trades, ct_init)

        # 손익비 & 손익분기 승률 계산
        rr = abs(ct_m['avg_win'] / ct_m['avg_loss']) if ct_m['avg_loss'] != 0 else float('inf')
        be_wr = 1 / (1 + rr) * 100 if rr != float('inf') else 0  # 손익분기 승률

        print(f"  완료 (CT {len(ct_trades)}건, PF={ct_m['pf']:.3f}, R/R={rr:.2f}, 손익분기WR={be_wr:.0f}%)")

        summary_rows.append({
            "설정":         name,
            "CT건수":       ct_m["n"],
            "CT승률(%)":    f"{ct_m['wr']:.1f}",
            "R/R":          f"{rr:.2f}" if rr != float('inf') else "∞",
            "손익분기WR":   f"{be_wr:.0f}%",
            "CT_PF":        f"{ct_m['pf']:.3f}" if ct_m['pf'] != float('inf') else "∞",
            "CT손익(USDT)": f"{ct_m['net_pnl']:+.1f}",
            "CT수익(%)":    f"{ct_m['net_pct']:+.1f}",
            "평균수익":     f"{ct_m['avg_win']:+.2f}",
            "평균손실":     f"{ct_m['avg_loss']:+.2f}",
            "DCA건":        ct_m["dca_trades"],
            "TP/SL/CTX":    f"{ct_m['tp_exits']}/{ct_m['sl_exits']}/{ct_m['ct_exits']}",
            "합산수익(%)":  f"{combined_pct:+.1f}",
        })

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CT 전략 비교 결과")
    print(f"{'='*70}")
    print(tabulate(summary_rows, headers="keys", tablefmt="simple"))

    # ── 퀀트 판단 기준 출력 ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  [퀀트 판단 기준]")
    print("  R/R (손익비)  : avg_win / |avg_loss|")
    print("  손익분기 승률: 1/(1+R/R)  →  실제승률 > 손익분기승률 이어야 수익")
    print("  PF > 1.0: 수익  |  PF > 1.2: 우수  |  PF > 1.5: 탁월")
    print("  합산수익이 가장 높은 설정이 메인 전략과의 시너지 최대")
    print(f"{'─'*70}")

    # 최우수 설정 추천
    best = max(
        summary_rows,
        key=lambda r: float(r["CT손익(USDT)"].replace("+", "").replace("∞", "0")),
    )
    print(f"\n  ★ CT 손익 최우수: [{best['설정']}]")
    print(f"    CT건수={best['CT건수']}  승률={best['CT승률(%)']}%"
          f"  손익={best['CT손익(USDT)']} USDT  PF={best['CT_PF']}")
    print(f"    청산사유 TP/SL/CTX = {best['TP/SL/CTX']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
