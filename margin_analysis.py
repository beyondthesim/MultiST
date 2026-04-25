"""
거래소 마진/레버리지 한도 점검

엔진 동작 분석:
  - entry_equity = USDT 명목 가치 (레버리지 1배 가정)
  - 메인 한 거래 명목 = $1,000 (현재 자본 전체)
  - CT 한 거래 명목 = ct_base_equity = $20,000 (ct_size_pct=2000)
  - CT는 DCA가 점진 진입이라 평균 명목은 작음

이 스크립트가 답하는 질문:
  1. CT의 실제 명목 포지션 분포 (DCA 발동률 반영)
  2. 메인 + CT 동시 보유 시 합산 명목 포지션
  3. 자본 $1,000으로 필요한 레버리지
  4. OKX 일반 알트(PI 같은) 한도와 비교
  5. 실거래 적용 가능 여부 + 권고
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate


def main():
    with open("config/params.json", encoding="utf-8") as f:
        params = json.load(f)

    safe_sym = params["symbol"].replace("/", "_").replace(":", "_")
    main_path = Path("results") / f"{safe_sym}_{params['timeframe']}_{params['start_date']}.csv"
    ct_path   = Path("results") / f"{safe_sym}_{params['timeframe']}_{params['start_date']}_ct.csv"

    main_df = pd.read_csv(main_path)
    ct_df   = pd.read_csv(ct_path)

    initial = params["initial_capital"]
    ct_cfg  = params["counter_trend"]

    print(f"\n{'='*80}")
    print(f"  거래소 마진/레버리지 한도 점검")
    print(f"  {params['symbol']} | 자본 ${initial:,.0f} | ct_size_pct={ct_cfg['ct_size_pct']}%")
    print(f"{'='*80}\n")

    # ── 1. 메인 거래 명목 포지션 분포 ─────────────────────────────────
    main_eq = main_df["entry_equity"]
    print("[1] 메인 거래 명목 포지션 분포 (USDT)")
    print(f"  거래수: {len(main_df)}")
    print(f"  최소:  ${main_eq.min():>10,.0f}")
    print(f"  평균:  ${main_eq.mean():>10,.0f}")
    print(f"  중앙:  ${main_eq.median():>10,.0f}")
    print(f"  최대:  ${main_eq.max():>10,.0f}")
    print(f"  p95:   ${main_eq.quantile(0.95):>10,.0f}")
    print(f"  → 자본 ${initial:,.0f} 대비 최대 {main_eq.max()/initial:.1f}x 레버리지 필요")

    # ── 2. CT 거래 명목 포지션 분포 ────────────────────────────────────
    ct_eq      = ct_df["entry_equity"]
    ct_dca     = ct_df["dca_count"] if "dca_count" in ct_df.columns else pd.Series([0]*len(ct_df))
    print(f"\n[2] CT 거래 명목 포지션 분포 (USDT)")
    print(f"  거래수: {len(ct_df)}")
    print(f"  최소:  ${ct_eq.min():>10,.0f}  (첫 진입만)")
    print(f"  평균:  ${ct_eq.mean():>10,.0f}")
    print(f"  중앙:  ${ct_eq.median():>10,.0f}")
    print(f"  최대:  ${ct_eq.max():>10,.0f}  (DCA {ct_dca.max()}회까지 들어간 경우)")
    print(f"  p95:   ${ct_eq.quantile(0.95):>10,.0f}")
    print(f"  → 최대 {ct_eq.max()/initial:.1f}x 레버리지 필요 (자본 ${initial:,.0f} 대비)")

    # ── 3. DCA 발동 분포 ──────────────────────────────────────────────
    print(f"\n[3] CT DCA 발동 분포")
    dca_dist = ct_dca.value_counts().sort_index()
    for n, cnt in dca_dist.items():
        pct = cnt / len(ct_df) * 100
        bar = "#" * int(pct / 2)
        print(f"  DCA {int(n):>2}회: {cnt:>3}건 ({pct:>5.1f}%) {bar}")

    # 이론상 최대 (DCA 10회 모두) vs 실제 최대
    dca_weights = ct_cfg.get("dca_weights", [])
    max_dca     = ct_cfg.get("max_dca", 0)
    total_w     = 1 + sum(dca_weights[:max_dca])
    per_unit    = (initial * ct_cfg["ct_size_pct"] / 100.0) / total_w
    theoretical_max = per_unit * total_w
    print(f"\n  per_unit (1 가중치): ${per_unit:.2f}")
    print(f"  이론상 최대 명목 (DCA {max_dca}회 모두): ${theoretical_max:,.0f}")
    print(f"  실제 발생한 최대 명목:                  ${ct_eq.max():,.0f}")
    print(f"  → 실제는 이론 최대의 {ct_eq.max()/theoretical_max*100:.1f}% 수준")

    # ── 4. 메인 + CT 동시 보유 명목 포지션 (시간 기반 추정) ────────────
    print(f"\n[4] 메인 + CT 동시 보유 시 합산 명목 포지션 추정")
    main_df["entry_time"] = pd.to_datetime(main_df["entry_time"])
    main_df["exit_time"]  = pd.to_datetime(main_df["exit_time"])
    ct_df["entry_time"]   = pd.to_datetime(ct_df["entry_time"])
    ct_df["exit_time"]    = pd.to_datetime(ct_df["exit_time"])

    overlaps = []
    for _, mt in main_df.iterrows():
        # 메인 거래 시간 동안 활성 CT 거래
        active_ct = ct_df[
            (ct_df["entry_time"] <= mt["exit_time"]) &
            (ct_df["exit_time"]  >= mt["entry_time"])
        ]
        if len(active_ct):
            combined = mt["entry_equity"] + active_ct["entry_equity"].sum()
            overlaps.append(combined)

    if overlaps:
        ov = np.array(overlaps)
        print(f"  메인-CT 시간 중첩 거래 쌍: {len(ov)}건")
        print(f"  합산 명목 평균:  ${ov.mean():>10,.0f}")
        print(f"  합산 명목 최대:  ${ov.max():>10,.0f}")
        print(f"  합산 명목 p95:   ${np.quantile(ov, 0.95):>10,.0f}")
        print(f"  → 자본 대비 최대 {ov.max()/initial:.1f}x 레버리지 필요")
    else:
        print(f"  메인-CT 동시 보유 사례 없음")

    # ── 5. 거래소 한도 비교 ────────────────────────────────────────────
    print(f"\n[5] OKX 일반 한도와 비교 (대략)")
    leverage_table = [
        ["BTC/ETH 영구",          "100x", "메이저, 가장 높음"],
        ["주요 알트(SOL/BNB 등)", "75x",  "유동성 좋음"],
        ["중간 알트",             "20~50x", "PI 같은 코인 보통 이 구간"],
        ["소형/신규 알트",        "5~10x",  "유동성 낮음"],
    ]
    print(tabulate(leverage_table, headers=["코인 등급", "최대 레버리지", "비고"], tablefmt="simple"))

    max_combined_lev = max(ov.max() / initial if overlaps else 0, ct_eq.max() / initial)
    print(f"\n  필요 레버리지: {max_combined_lev:.1f}x (메인+CT 합산 최대)")
    if max_combined_lev <= 10:
        verdict = "[가능] 안전 - 대부분 거래소 한도 내"
    elif max_combined_lev <= 20:
        verdict = "[주의] PI 같은 알트 한도 근접 - 거래소별 확인 필요"
    elif max_combined_lev <= 50:
        verdict = "[위험] 한도 초과 가능 - 큰 알트만 가능"
    else:
        verdict = "[불가] 대부분 거래소 불가 - 자본 증액 필요"
    print(f"  판정: {verdict}")

    # ── 6. 자본 시나리오 ──────────────────────────────────────────────
    print(f"\n[6] 자본 시나리오별 필요 레버리지")
    needed_notional = max(ov.max() if overlaps else 0, ct_eq.max())
    scenarios = [1000, 2000, 5000, 10000, 20000, 50000]
    rows = []
    for cap in scenarios:
        lev_needed = needed_notional / cap
        ok = "OK" if lev_needed <= 10 else ("주의" if lev_needed <= 20 else "위험")
        rows.append([
            f"${cap:,}",
            f"${needed_notional:,.0f}",
            f"{lev_needed:.1f}x",
            ok,
        ])
    print(tabulate(rows, headers=["자본", "최대 명목", "필요 레버리지", "판정"], tablefmt="simple"))

    # ── 7. 슬리피지 추정 ──────────────────────────────────────────────
    print(f"\n[7] 슬리피지 위험 (PI 같은 작은 알트)")
    print(f"  CT 최대 단일 명목: ${ct_eq.max():,.0f}")
    print(f"  PI 일평균 거래량 가정 ~수억$ 수준이면, ${ct_eq.max():,.0f}는")
    print(f"  → 시장가 진입 시 0.05~0.2% 슬리피지 예상")
    print(f"  → 1년 88건 × 평균 0.1% = ~$1,760 추가 비용 (현재 +$1,521 이익이 침식될 수 있음)")

    # ── 8. 권고 ───────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  [최종 권고]")
    print(f"  현재 시뮬: 자본 $1,000으로 최대 {max_combined_lev:.0f}x 레버리지 필요")
    print(f"  → 실거래 적용 시 다음 중 택일:")
    print(f"    A. 자본을 ${needed_notional/10:,.0f}~${needed_notional/5:,.0f}로 증액 (10~20x 안전대)")
    print(f"    B. ct_size_pct를 200~500%로 축소 (사이즈 x2~x5)")
    print(f"    C. 거래소 등급 확인 후 PI 한도가 20x 이상이면 자본 ${needed_notional/20:,.0f}+ 로 운용")
    print(f"  → 슬리피지 누적 비용 고려해 수익 예상 -10~-20% 보정 권장")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
