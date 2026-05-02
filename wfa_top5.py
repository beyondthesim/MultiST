"""
Walk-Forward Analysis: 매일 상위 5개 코인 선별 → 다음날 균등 투자

설계:
1. 모든 코인의 백테스트를 한 번씩 실행 → 일별 자본곡선 추출
2. 매일 룩백 N일 누적 수익률 기준 상위 5 선별
3. 다음 1일은 그 5개의 평균 일일 수익률을 포트폴리오에 반영 (균등 20%씩)
4. 자본 이월하며 끝까지 진행

사용법:
    python wfa_top5.py
    python wfa_top5.py --lookback 3 --topn 5 --tf 12m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("C:/trade/DynamicDCAHedge/data/raw")
BTC_FILE = RAW_DIR / "BTC_USDT_USDT_4h.parquet"

_RESAMPLE_ALIAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "12m": "12min",
    "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h",
}


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    alias = _RESAMPLE_ALIAS[tf]
    return (
        df[["open", "high", "low", "close", "volume"]]
        .resample(alias, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )


def _symbol_label(path: Path) -> str:
    return path.stem.replace("_1m", "").split("_")[0]


def _build_eq_1m_from_trades(
    main_trades: list[dict],
    ct_trades: list[dict],
    df_1m: pd.DataFrame,
    initial_cap: float,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
) -> pd.Series:
    """
    Method B (정확 버전): trades 정보 + 1m close로 1m 자본곡선 재구성.

    각 1m 시점에서:
      활성 trade(entry_time <= t < exit_time)의 unrealized PnL +
      청산된 trade(exit_time <= t)의 net_pnl 누적
    = initial + closed_pnl + unrealized

    단순화:
      - entry_price = 최종 평균가 (DCA 후) — 진입 직후 약간 부정확
      - 부분청산 무시 (exit_time에 일괄 청산 처리)
    """
    mask = (df_1m.index >= bt_start) & (df_1m.index < bt_end)
    idx = df_1m.index[mask]
    close = df_1m.loc[mask, "close"].values
    n = len(idx)
    if n == 0:
        return pd.Series([initial_cap], index=[bt_start])

    closed = np.zeros(n)
    unrealized = np.zeros(n)

    all_trades = (main_trades or []) + (ct_trades or [])
    for t in all_trades:
        if t.get("exit_time") is None:
            continue
        d_str = t["direction"]
        d = 1 if d_str in ("LONG", "long", 1) else -1
        ep = float(t["entry_price"])
        eq_in = float(t["entry_equity"])
        net = float(t["net_pnl"])
        et = pd.Timestamp(t["entry_time"])
        xt = pd.Timestamp(t["exit_time"])
        if ep <= 0:
            continue
        units = eq_in / ep

        # 활성 구간 [et, xt)
        active = (idx >= et) & (idx < xt)
        if active.any():
            unrealized[active] += (close[active] - ep) * d * units

        # 청산 후
        after = idx >= xt
        if after.any():
            closed[after] += net

    eq = initial_cap + closed + unrealized
    s = pd.Series(eq, index=idx)
    s = s[~s.index.duplicated(keep="last")]
    return s


def _backtest_one(
    symbol: str,
    df_1m: pd.DataFrame,
    btc_df,
    params: dict,
    tf: str,
    start_offset_days: int,
    end_offset_days: int,
    min_bt_days: int,
    main_tf: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """
    한 코인 백테스트 후 (일별 자본, 일별 수익률, 봉(TF) 자본, 1m 자본) 반환.
    main_tf가 None이면 단일 TF 모드, 지정 시 split TF 모드.
    """
    from strategy.signal import build_signals, build_signals_split_tf
    from backtest.engine import BacktestEngine

    if df_1m.empty:
        return None

    df_tf = _resample(df_1m, tf)
    if df_tf.empty:
        return None

    listing_dt = df_tf.index[0]
    end_data = df_tf.index[-1]
    bt_start_dt = listing_dt + pd.Timedelta(days=start_offset_days)
    bt_end_dt = min(end_data, listing_dt + pd.Timedelta(days=end_offset_days))

    if bt_end_dt - bt_start_dt < pd.Timedelta(days=min_bt_days):
        return None

    df_tf = df_tf[df_tf.index < bt_end_dt].copy()
    df_tf["in_backtest"] = df_tf.index >= bt_start_dt

    if df_tf["in_backtest"].sum() < 50:
        return None

    p = dict(params)
    p["symbol"] = f"{symbol}/USDT:USDT"
    # split 모드: 메인 TF는 main_tf, base/CT TF는 tf
    p["timeframe"] = main_tf if main_tf else tf

    try:
        if main_tf and main_tf != tf:
            df_sig = build_signals_split_tf(df_tf, p, main_tf=main_tf, btc_df=btc_df)
        else:
            df_sig = build_signals(df_tf, p, btc_df=btc_df)
        df_bt = df_sig[df_sig["in_backtest"]].copy()
        if df_bt.empty:
            return None

        engine = BacktestEngine(p)
        main_t, ct_t, eq_curve = engine.run(df_bt)
    except Exception as e:
        print(f"  [{symbol}] 엔진 오류: {e}")
        return None

    if not eq_curve:
        return None

    init_cap = p["initial_capital"]
    eq_df = pd.DataFrame(eq_curve)
    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
    bar_eq = eq_df.set_index("timestamp")["equity"]

    # 1m 자본곡선 재구성 (Method B 정확 버전 - trades 기반)
    bar_eq_1m = _build_eq_1m_from_trades(
        main_t, ct_t, df_1m, init_cap, bt_start_dt, bt_end_dt
    )

    # 일별 마지막 값 (UTC 자정 기준)
    daily_eq = bar_eq.resample("1D", label="left", closed="left").last().dropna()
    if len(daily_eq) < 2:
        return None

    daily_eq = pd.concat(
        [pd.Series([init_cap], index=[daily_eq.index[0] - pd.Timedelta(days=1)]),
         daily_eq]
    )
    daily_ret = daily_eq.pct_change().dropna()
    return daily_eq, daily_ret, bar_eq, bar_eq_1m


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--tf", default="12m",
                    help="base TF (split 모드에선 CT TF, 단일 모드에선 메인=CT TF)")
    ap.add_argument("--main-tf", default=None,
                    help="메인 전략 TF (지정 시 split TF 모드 활성: base=tf, 메인=main_tf)")
    ap.add_argument("--start-offset", type=int, default=7)
    ap.add_argument("--end-offset", type=int, default=60)
    ap.add_argument("--min-bt-days", type=int, default=14)
    ap.add_argument("--lookback", type=int, default=7,
                    help="상위 N 선별을 위한 룩백 일수")
    ap.add_argument("--topn", type=int, default=5,
                    help="상위 몇 개 선별")
    ap.add_argument("--initial", type=float, default=1000)
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        params = json.load(f)

    btc_df = None
    if params.get("btc_filter", {}).get("enabled", False) and BTC_FILE.exists():
        btc_df = pd.read_parquet(BTC_FILE)
        btc_df.index = pd.to_datetime(btc_df.index, utc=True)
        print(f"BTC 매크로 로드: {len(btc_df):,}봉")

    files = sorted(RAW_DIR.glob("*_USDT_USDT_1m.parquet"))
    print(f"\n=== WFA TOP{args.topn} 백테스트 ===")
    print(f"심볼 수: {len(files)}  TF: {args.tf}")
    print(f"룩백: {args.lookback}일  TopN: {args.topn}  자본분배: 균등")
    print()

    # 1단계: 모든 코인 일별/봉 단위 자본 수집
    daily_returns: dict[str, pd.Series] = {}
    daily_equities: dict[str, pd.Series] = {}
    bar_equities: dict[str, pd.Series] = {}
    bar_equities_1m: dict[str, pd.Series] = {}

    skipped = 0
    for i, f in enumerate(files, 1):
        sym = _symbol_label(f)
        print(f"[{i:3d}/{len(files)}] {sym:14s}  ", end="", flush=True)

        df_1m = pd.read_parquet(f)
        df_1m.index = pd.to_datetime(df_1m.index, utc=True)

        result = _backtest_one(
            symbol=sym, df_1m=df_1m, btc_df=btc_df, params=params,
            tf=args.tf,
            start_offset_days=args.start_offset,
            end_offset_days=args.end_offset,
            min_bt_days=args.min_bt_days,
            main_tf=args.main_tf,
        )
        if result is None:
            skipped += 1
            print("SKIP")
            continue

        daily_eq, daily_ret, bar_eq, bar_eq_1m = result
        daily_returns[sym] = daily_ret
        daily_equities[sym] = daily_eq
        bar_equities[sym] = bar_eq
        bar_equities_1m[sym] = bar_eq_1m
        print(f"{len(daily_ret)}일/{len(bar_eq)}TF/{len(bar_eq_1m)}1m  최종 자본: {daily_eq.iloc[-1]:.0f}")

    if not daily_returns:
        print("결과 없음")
        return

    # 2단계: 일별 수익률 매트릭스 (행=날짜, 열=코인)
    ret_mat = pd.DataFrame(daily_returns)
    ret_mat.index = pd.to_datetime(ret_mat.index, utc=True).normalize()
    ret_mat = ret_mat.sort_index()

    print(f"\n수집 완료: {len(daily_returns)}개 코인, "
          f"{skipped}개 스킵, "
          f"{len(ret_mat)}일 ({ret_mat.index.min().date()} ~ {ret_mat.index.max().date()})")

    # 3단계: WFA 시뮬레이션
    # 누적 수익률(cumprod-1)로 룩백 평가
    initial = args.initial
    portfolio = [initial]
    portfolio_dates = [ret_mat.index[0]]
    daily_picks: list[dict] = []

    # 룩백 기간 후부터 시작
    # 추가 보수: TopN개 이상의 코인이 룩백 풀데이터 가질 수 있는 시점부터 시작
    valid_dates = ret_mat.index[ret_mat.index >= (ret_mat.index[0] + pd.Timedelta(days=args.lookback))]

    if len(valid_dates) == 0:
        print(f"룩백 {args.lookback}일 후 가용 일자 없음 → 룩백 단축 권장")
        return

    # 가장 빠른 안전 시작점 찾기: TopN개 코인이 룩백 N일 풀데이터 + cur_date 데이터 있는 시점
    safe_start = None
    for d in valid_dates:
        lb_start = d - pd.Timedelta(days=args.lookback)
        lb_win = ret_mat[(ret_mat.index >= lb_start) & (ret_mat.index < d)]
        if len(lb_win) < args.lookback:
            continue
        full_d = lb_win.notna().sum() == len(lb_win)
        elig = [s for s in full_d[full_d].index if pd.notna(ret_mat.loc[d, s])]
        if len(elig) >= args.topn:
            safe_start = d
            break

    if safe_start is None:
        print(f"안전 시작점 없음 (TopN={args.topn} 후보 미확보)")
        return

    valid_dates = valid_dates[valid_dates >= safe_start]
    print(f"안전 시작일: {safe_start.date()} (lookback {args.lookback}일 × top{args.topn} 후보 확보)")

    # ─ Lookahead-bias 방지 엄격 규칙 ─
    # 1) 룩백 윈도우: ret_mat.index < cur_date  (cur_date 자체 제외)
    # 2) 후보 자격: 룩백 N일 모두 실제 데이터 있어야 함 (NaN 0으로 채우지 않음)
    # 3) cur_date에도 데이터가 있어야 그 날 사용 (NaN 코인 자동 제외)
    # 4) WFA 시작: cur_date - lookback 이전에도 모든 후보가 적어도 한 명 이상 존재해야 의미 있음
    skipped_no_candidates = 0

    for cur_date in valid_dates:
        # 룩백 윈도우: [cur_date - lookback, cur_date)
        lookback_start = cur_date - pd.Timedelta(days=args.lookback)
        lookback_window = ret_mat[
            (ret_mat.index >= lookback_start) & (ret_mat.index < cur_date)
        ]
        if lookback_window.empty or len(lookback_window) < args.lookback:
            # 룩백 일수 미만이면 시뮬 미수행 (자본 유지)
            portfolio.append(portfolio[-1])
            portfolio_dates.append(cur_date)
            continue

        # ★ 엄격 규칙: 룩백 전체 N일 모두 실제 데이터 있어야 후보 자격
        full_data = lookback_window.notna().sum() == len(lookback_window)
        eligible = full_data[full_data].index

        # ★ cur_date 시점에도 실제 데이터 있어야 함 (그 날 트레이드 가능)
        eligible = [s for s in eligible if pd.notna(ret_mat.loc[cur_date, s])]

        if not eligible:
            skipped_no_candidates += 1
            portfolio.append(portfolio[-1])
            portfolio_dates.append(cur_date)
            continue

        # 룩백 누적 수익률 (NaN 채움 없음, 후보 코인은 모두 N일 데이터 있음)
        cum_ret = (1 + lookback_window[eligible]).prod() - 1
        sorted_syms = cum_ret.sort_values(ascending=False).index.tolist()
        top_syms = sorted_syms[: args.topn]

        # 다음날(=cur_date) 수익률을 사용 (균등 가중)
        today_returns = ret_mat.loc[cur_date, top_syms]
        # 모든 top_syms은 위에서 이미 cur_date 데이터 있다고 보장됨
        day_ret = float(today_returns.mean())

        new_eq = portfolio[-1] * (1 + day_ret)
        portfolio.append(new_eq)
        portfolio_dates.append(cur_date)

        daily_picks.append({
            "date": cur_date.strftime("%Y-%m-%d"),
            "n_eligible": len(eligible),
            "picks": ",".join(top_syms),
            "day_ret_pct": round(day_ret * 100, 3),
            "equity": round(new_eq, 2),
        })

    if skipped_no_candidates:
        print(f"  (후보 없는 일자 {skipped_no_candidates}개 - 자본 유지)")

    # 4단계: 봉 단위 자본곡선 합성 (진짜 MDD 측정용)
    # 각 cur_date에 결정된 top_syms 5개의 봉 단위 수익률을
    # 그 24시간 동안 균등 가중 평균으로 포트폴리오 자본에 적용
    bar_ret_mat = pd.DataFrame({s: e.pct_change() for s, e in bar_equities.items()})
    bar_ret_mat = bar_ret_mat.sort_index()

    # 수수료 옵션: 매일 5개 중 바뀐 코인만 청산-진입 비용
    # 보수적: 종목 1개당 0.05% × 2(in/out) = 0.10%, 비중 1/5 = 0.02%/회전
    commission_per_swap = 0.0005 * 2 / args.topn  # 0.02% per swap

    bar_eq_curve: list[tuple[pd.Timestamp, float]] = []
    cur_eq = initial
    bar_eq_curve.append((portfolio_dates[0], cur_eq))
    prev_picks: set = set()

    for pick_info in daily_picks:
        cur_date = pd.Timestamp(pick_info["date"], tz="UTC")
        next_date = cur_date + pd.Timedelta(days=1)
        top_syms = pick_info["picks"].split(",")

        # 종목 회전 수수료 (어제와 다른 코인만)
        new_picks = set(top_syms) - prev_picks
        n_swaps = len(new_picks)
        if n_swaps > 0 and prev_picks:
            cur_eq *= (1 - commission_per_swap * n_swaps)

        # 그 24시간 봉 단위 수익률 (top_syms 균등 평균)
        mask = (bar_ret_mat.index >= cur_date) & (bar_ret_mat.index < next_date)
        bars_window = bar_ret_mat.loc[mask, top_syms]
        # 각 봉별 평균 (NaN은 그 코인 봉이 없는 경우 → 무시)
        avg_bar = bars_window.mean(axis=1, skipna=True).fillna(0)

        for ts, r in avg_bar.items():
            cur_eq *= (1 + r)
            bar_eq_curve.append((ts, cur_eq))

        prev_picks = set(top_syms)

    bar_eq_series = pd.Series(
        [v for _, v in bar_eq_curve],
        index=[t for t, _ in bar_eq_curve],
    ).sort_index()

    # 봉 단위 MDD
    bar_peak = bar_eq_series.cummax()
    bar_dd = (bar_eq_series - bar_peak) / bar_peak * 100
    bar_mdd = float(bar_dd.min())
    bar_final = float(bar_eq_series.iloc[-1])
    bar_total_ret = (bar_final - initial) / initial * 100

    # ─ Method B: 1m 자본곡선 합성 ─────────────────────────────────────
    # 각 코인의 1m eq를 pct_change로 변환, top_syms 균등 평균을 매분 적용
    bar_ret_1m_mat = pd.DataFrame(
        {s: e.pct_change() for s, e in bar_equities_1m.items()}
    ).sort_index()

    bar_eq_1m_curve: list[tuple[pd.Timestamp, float]] = []
    cur_eq_1m = initial
    bar_eq_1m_curve.append((portfolio_dates[0], cur_eq_1m))
    prev_picks_1m: set = set()

    for pick_info in daily_picks:
        cur_date = pd.Timestamp(pick_info["date"], tz="UTC")
        next_date = cur_date + pd.Timedelta(days=1)
        top_syms = pick_info["picks"].split(",")

        new_picks = set(top_syms) - prev_picks_1m
        n_swaps = len(new_picks)
        if n_swaps > 0 and prev_picks_1m:
            cur_eq_1m *= (1 - commission_per_swap * n_swaps)

        mask = (bar_ret_1m_mat.index >= cur_date) & (bar_ret_1m_mat.index < next_date)
        bars_window = bar_ret_1m_mat.loc[mask, top_syms]
        avg_bar = bars_window.mean(axis=1, skipna=True).fillna(0)

        for ts, r in avg_bar.items():
            cur_eq_1m *= (1 + r)
            bar_eq_1m_curve.append((ts, cur_eq_1m))

        prev_picks_1m = set(top_syms)

    bar_eq_1m_series = pd.Series(
        [v for _, v in bar_eq_1m_curve],
        index=[t for t, _ in bar_eq_1m_curve],
    ).sort_index()

    bar_1m_peak = bar_eq_1m_series.cummax()
    bar_1m_dd = (bar_eq_1m_series - bar_1m_peak) / bar_1m_peak * 100
    bar_1m_mdd = float(bar_1m_dd.min())
    bar_1m_final = float(bar_eq_1m_series.iloc[-1])
    bar_1m_total_ret = (bar_1m_final - initial) / initial * 100

    eq_series = pd.Series(portfolio, index=portfolio_dates)

    final_eq = eq_series.iloc[-1]
    total_ret_pct = (final_eq - initial) / initial * 100

    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100
    mdd = float(dd.min())

    daily_rets_pf = eq_series.pct_change().dropna()
    win_days = (daily_rets_pf > 0).sum()
    total_days = len(daily_rets_pf)
    win_rate_d = win_days / total_days * 100 if total_days else 0
    avg_daily = daily_rets_pf.mean() * 100
    sharpe = (
        daily_rets_pf.mean() / daily_rets_pf.std() * np.sqrt(365)
        if daily_rets_pf.std() > 0 else 0
    )

    print("\n" + "=" * 70)
    print(f"  WFA 결과 (룩백 {args.lookback}일, TOP{args.topn}, 균등 분배)")
    print("=" * 70)
    print(f"기간: {eq_series.index[0].date()} ~ {eq_series.index[-1].date()} ({total_days}일)")
    print(f"초기 자본: {initial:.0f}")
    print()
    print(f"[일별 종가 기준 - 분산효과로 MDD 과소평가 가능]")
    print(f"  최종 자본: {final_eq:.2f}  총 수익률: {total_ret_pct:+.2f}%")
    print(f"  일승률: {win_rate_d:.1f}% ({win_days}/{total_days})")
    print(f"  MDD(일별): {mdd:.2f}%   샤프(연환산): {sharpe:.2f}")
    print()
    print(f"[봉 단위({args.tf}) 기준 - 회전 수수료 {commission_per_swap*100:.3f}%/swap 반영]")
    print(f"  최종 자본: {bar_final:.2f}  총 수익률: {bar_total_ret:+.2f}%")
    print(f"  MDD({args.tf}): {bar_mdd:.2f}%   봉 수: {len(bar_eq_series):,}")
    print()
    print(f"[Method B: 1분봉 인터폴레이션 기준 - 인트라봉 변동성 반영]")
    print(f"  최종 자본: {bar_1m_final:.2f}  총 수익률: {bar_1m_total_ret:+.2f}%")
    print(f"  MDD(1m): {bar_1m_mdd:.2f}%   봉 수: {len(bar_eq_1m_series):,}")

    # 매일 선별된 코인 빈도 통계
    print("\n" + "=" * 70)
    print("  코인별 선별 빈도 TOP 15")
    print("=" * 70)
    pick_counts: dict[str, int] = {}
    for p in daily_picks:
        for s in p["picks"].split(","):
            pick_counts[s] = pick_counts.get(s, 0) + 1
    pick_df = pd.DataFrame(
        sorted(pick_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["symbol", "pick_days"]
    )
    pick_df["pick_pct"] = (pick_df["pick_days"] / total_days * 100).round(1)
    print(pick_df.head(15).to_string(index=False))

    # 일자별 상세 (마지막 10일)
    print("\n" + "=" * 70)
    print("  최근 10일 선별 내역")
    print("=" * 70)
    df_picks = pd.DataFrame(daily_picks)
    print(df_picks.tail(10).to_string(index=False))

    # 비교: 모든 코인 균등 분배 (벤치마크)
    bench_daily = ret_mat.fillna(0).mean(axis=1)
    bench_eq = (1 + bench_daily).cumprod() * initial
    bench_final = float(bench_eq.iloc[-1])
    bench_ret = (bench_final - initial) / initial * 100
    bench_peak = bench_eq.cummax()
    bench_dd = (bench_eq - bench_peak) / bench_peak * 100
    bench_mdd = float(bench_dd.min())

    print("\n" + "=" * 70)
    print("  벤치마크 비교")
    print("=" * 70)
    print(f"WFA TOP{args.topn} (룩백 {args.lookback}일) | 수익 {total_ret_pct:+7.2f}%  MDD {mdd:6.2f}%  Sharpe {sharpe:.2f}")
    print(f"전체 균등 분배 (101개 모두) | 수익 {bench_ret:+7.2f}%  MDD {bench_mdd:6.2f}%")

    # CSV 저장
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    eq_df = pd.DataFrame({"date": eq_series.index, "equity": eq_series.values})
    eq_path = out_dir / f"wfa_top{args.topn}_lb{args.lookback}_{today}_equity.csv"
    eq_df.to_csv(eq_path, index=False)

    if daily_picks:
        pick_path = out_dir / f"wfa_top{args.topn}_lb{args.lookback}_{today}_picks.csv"
        pd.DataFrame(daily_picks).to_csv(pick_path, index=False)
        print(f"\n저장: {eq_path}")
        print(f"저장: {pick_path}")


if __name__ == "__main__":
    main()
