"""
신규상장 코인 백테스트 스캔
- C:/trade/DynamicDCAHedge/data/raw/ 에 있는 1분봉 parquet를 읽어
  목표 타임프레임으로 리샘플링 후 백테스트
- 각 코인의 데이터 시작일을 "유사 상장일"로 간주
- 윈도우: [시작일+offset_start_days, min(시작일+offset_end_days, 데이터 끝)]
- 메인 추세 / CT 역추세 결과를 분리 집계

사용법:
    python scan_newlistings.py
    python scan_newlistings.py --tf 12m --start-offset 7 --end-offset 60
    python scan_newlistings.py --min-bt-days 14   # 최소 백테스트 일수
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

RAW_DIR = Path("C:/trade/DynamicDCAHedge/data/raw")
BTC_FILE = RAW_DIR / "BTC_USDT_USDT_4h.parquet"

_RESAMPLE_ALIAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "12m": "12min",
    "15m": "15min", "30m": "30min", "45m": "45min",
    "1h": "1h", "2h": "2h", "4h": "4h",
}


def _tf_minutes(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"unknown tf: {tf}")


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    alias = _RESAMPLE_ALIAS.get(tf)
    if alias is None:
        raise ValueError(f"unsupported tf: {tf}")
    out = (
        df[["open", "high", "low", "close", "volume"]]
        .resample(alias, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    return out


def _list_symbols() -> list[Path]:
    files = sorted(RAW_DIR.glob("*_USDT_USDT_1m.parquet"))
    return files


def _symbol_label(path: Path) -> str:
    base = path.stem.replace("_1m", "")
    parts = base.split("_")
    return parts[0]


@dataclass
class WindowResult:
    symbol: str
    listing_date: str
    bt_start: str
    bt_end: str
    bt_days: int
    bars: int
    # 메인 (trades 기반)
    main_trades: int
    main_wr: float
    main_profit_pct: float
    main_pf: float
    main_mdd: float
    # CT (trades 기반 — DCA 중 미실현 손실 미반영)
    ct_trades: int
    ct_wr: float
    ct_profit_pct: float
    ct_pf: float
    ct_mdd_trade: float
    # 합산 (equity_curve 기반 — 미실현 포함, 가장 현실적)
    combined_profit_pct: float
    combined_mdd: float


def _stats(trades: list[dict], init_cap: float) -> dict:
    """trades만으로 통계 계산 (main/ct 분리용). MDD는 trades cumulative pnl 기반."""
    if not trades:
        return {"trades": 0, "wr": 0.0, "profit_pct": 0.0, "pf": 0.0, "mdd": 0.0}

    n = len(trades)
    winners = sum(1 for t in trades if t["net_pnl"] > 0)
    wr = winners / n * 100

    gp = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gl = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    pf = gp / gl if gl > 0 else float("inf")

    total_pnl = sum(t["net_pnl"] for t in trades)
    profit_pct = total_pnl / init_cap * 100

    # MDD: 시간순 trades pnl 누적 곡선 기준
    def _ts(t):
        v = t.get("exit_time") or t.get("entry_time")
        return v if v is not None else pd.Timestamp.min.tz_localize("UTC")

    sorted_t = sorted(trades, key=_ts)
    eq = init_cap
    peak = init_cap
    mdd = 0.0
    for t in sorted_t:
        eq += t["net_pnl"]
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd

    return {
        "trades": n,
        "wr": round(wr, 1),
        "profit_pct": round(profit_pct, 2),
        "pf": round(pf, 3) if pf != float("inf") else 999.0,
        "mdd": round(mdd, 1),
    }


def run_one(
    symbol: str,
    df_1m: pd.DataFrame,
    btc_df: Optional[pd.DataFrame],
    params: dict,
    tf: str,
    start_offset_days: int,
    end_offset_days: int,
    warmup_bars: int,
    min_bt_days: int,
) -> Optional[WindowResult]:
    from strategy.signal import build_signals
    from backtest.engine import BacktestEngine

    if df_1m.empty:
        return None

    # 1m → tf 리샘플
    df_tf = _resample(df_1m, tf)
    if df_tf.empty:
        return None

    listing_dt = df_tf.index[0]
    end_data = df_tf.index[-1]

    bt_start_dt = listing_dt + pd.Timedelta(days=start_offset_days)
    bt_end_dt = min(end_data, listing_dt + pd.Timedelta(days=end_offset_days))

    if bt_end_dt - bt_start_dt < pd.Timedelta(days=min_bt_days):
        return None

    # in_backtest 마킹 (워밍업: bt_start 직전까지)
    df_tf = df_tf[df_tf.index < bt_end_dt].copy()
    df_tf["in_backtest"] = df_tf.index >= bt_start_dt

    if df_tf["in_backtest"].sum() < 50:
        return None

    p = dict(params)
    p["symbol"] = f"{symbol}/USDT:USDT"
    p["timeframe"] = tf

    try:
        df_sig = build_signals(df_tf, p, btc_df=btc_df)
        df_bt = df_sig[df_sig["in_backtest"]].copy()
        if df_bt.empty:
            return None

        engine = BacktestEngine(p)
        main_trades, ct_trades, eq_curve = engine.run(df_bt)
    except Exception as e:
        print(f"  [{symbol}] 엔진 오류: {e}")
        return None

    init_cap = p["initial_capital"]
    main_stats = _stats(main_trades, init_cap)
    ct_stats = _stats(ct_trades, init_cap)

    # 합산 통계: equity_curve(미실현 포함) 기반
    if eq_curve:
        final_eq = eq_curve[-1]["equity"]
        combined_profit_pct = (final_eq - init_cap) / init_cap * 100
        peak = init_cap
        cmdd = 0.0
        for pt in eq_curve:
            eq = pt["equity"]
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100
                if dd > cmdd:
                    cmdd = dd
    else:
        combined_profit_pct = 0.0
        cmdd = 0.0

    bars = int(df_bt["in_backtest"].sum())
    bt_days = (bt_end_dt - bt_start_dt).days

    return WindowResult(
        symbol=symbol,
        listing_date=listing_dt.strftime("%Y-%m-%d"),
        bt_start=bt_start_dt.strftime("%Y-%m-%d"),
        bt_end=bt_end_dt.strftime("%Y-%m-%d"),
        bt_days=bt_days,
        bars=bars,
        main_trades=main_stats["trades"],
        main_wr=main_stats["wr"],
        main_profit_pct=main_stats["profit_pct"],
        main_pf=main_stats["pf"],
        main_mdd=main_stats["mdd"],
        ct_trades=ct_stats["trades"],
        ct_wr=ct_stats["wr"],
        ct_profit_pct=ct_stats["profit_pct"],
        ct_pf=ct_stats["pf"],
        ct_mdd_trade=ct_stats["mdd"],
        combined_profit_pct=round(combined_profit_pct, 2),
        combined_mdd=round(cmdd, 1),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="신규상장 코인 백테스트 스캔")
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--tf", default="12m", help="목표 타임프레임 (기본 12m)")
    ap.add_argument("--start-offset", type=int, default=7,
                    help="상장일 기준 백테스트 시작 오프셋(일) (기본 7)")
    ap.add_argument("--end-offset", type=int, default=60,
                    help="상장일 기준 백테스트 종료 오프셋(일) (기본 60)")
    ap.add_argument("--min-bt-days", type=int, default=14,
                    help="최소 백테스트 일수 (이보다 짧으면 스킵, 기본 14)")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="특정 심볼만 (예: PNUT BIO)")
    ap.add_argument("--top", type=int, default=30, help="출력 상위 N (기본 30)")
    ap.add_argument("--sort", default="combined_profit_pct",
                    help="정렬 컬럼 (기본 combined_profit_pct)")
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        params = json.load(f)

    # BTC 매크로 필터 데이터 (외부 캐시 사용)
    btc_df = None
    btc_cfg = params.get("btc_filter", {})
    if btc_cfg.get("enabled", False):
        if BTC_FILE.exists():
            btc_df = pd.read_parquet(BTC_FILE)
            btc_df.index = pd.to_datetime(btc_df.index, utc=True)
            print(f"BTC 매크로 필터 로드: {len(btc_df):,}봉  "
                  f"({btc_df.index[0].date()} ~ {btc_df.index[-1].date()})")
        else:
            print(f"WARN: BTC 매크로 데이터 없음 → 필터 비활성화")
            params["btc_filter"]["enabled"] = False

    files = _list_symbols()
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        files = [f for f in files if _symbol_label(f).upper() in wanted]

    print(f"\n=== 신규상장 코인 스캔 ===")
    print(f"심볼 수: {len(files)}")
    print(f"타임프레임: {args.tf}")
    print(f"윈도우: 상장일+{args.start_offset}일 ~ 상장일+{args.end_offset}일")
    print(f"최소 백테스트 일수: {args.min_bt_days}\n")

    warmup_bars = 200
    results: list[WindowResult] = []
    skipped = 0

    for i, f in enumerate(files, 1):
        sym = _symbol_label(f)
        print(f"[{i:3d}/{len(files)}] {sym:14s}  ", end="", flush=True)

        df_1m = pd.read_parquet(f)
        df_1m.index = pd.to_datetime(df_1m.index, utc=True)

        r = run_one(
            symbol=sym,
            df_1m=df_1m,
            btc_df=btc_df,
            params=params,
            tf=args.tf,
            start_offset_days=args.start_offset,
            end_offset_days=args.end_offset,
            warmup_bars=warmup_bars,
            min_bt_days=args.min_bt_days,
        )
        if r is None:
            skipped += 1
            print("SKIP (데이터 부족)")
            continue

        print(f"main:{r.main_trades:3d}건 {r.main_profit_pct:+7.2f}%"
              f" | ct:{r.ct_trades:3d}건 {r.ct_profit_pct:+6.2f}%"
              f" | 합산:{r.combined_profit_pct:+7.2f}% MDD{r.combined_mdd:5.1f}%"
              f"  ({r.bt_days}일)")
        results.append(r)

    if not results:
        print("\n결과 없음.")
        return

    print(f"\n완료: {len(results)}개 백테스트, {skipped}개 스킵\n")

    df_res = pd.DataFrame([r.__dict__ for r in results])

    sort_col = args.sort if args.sort in df_res.columns else "combined_profit_pct"

    # ── 합산 결과 (메인 + CT 통합 자본곡선 기반, 미실현 포함) ─────────
    print("=" * 120)
    print(f"  [메인 + CT 합산 결과] equity_curve 기반, 미실현 손실 포함 (가장 현실적)")
    print("=" * 120)
    df_combined = df_res.sort_values(sort_col, ascending=False)
    show_cols_c = ["symbol", "listing_date", "bt_days",
                   "main_trades", "main_profit_pct",
                   "ct_trades", "ct_profit_pct",
                   "combined_profit_pct", "combined_mdd"]
    print(df_combined[show_cols_c].head(args.top).to_string(index=False))

    print("\n" + "=" * 120)
    print(f"  메인 추세 전략 단독 결과")
    print("=" * 120)
    df_main = df_res.sort_values("main_profit_pct", ascending=False)
    show_cols_m = ["symbol", "listing_date", "bt_days", "main_trades",
                   "main_wr", "main_profit_pct", "main_pf", "main_mdd"]
    print(df_main[show_cols_m].head(args.top).to_string(index=False))

    print("\n" + "=" * 120)
    print(f"  CT 역추세 전략 단독 결과 (주의: ct_mdd_trade는 trades 기반이라 DCA 미실현 손실 미반영)")
    print("=" * 120)
    df_ct = df_res.sort_values("ct_profit_pct", ascending=False)
    show_cols_ct = ["symbol", "listing_date", "bt_days", "ct_trades",
                    "ct_wr", "ct_profit_pct", "ct_pf", "ct_mdd_trade"]
    print(df_ct[show_cols_ct].head(args.top).to_string(index=False))

    # ── 합산 통계 ────────────────────────────────────────────────────
    n = len(df_res)
    main_pos = (df_res["main_profit_pct"] > 0).sum()
    ct_pos = (df_res["ct_profit_pct"] > 0).sum()
    combined_pos = (df_res["combined_profit_pct"] > 0).sum()

    print("\n" + "=" * 70)
    print("  요약 통계")
    print("=" * 70)
    print(f"전체 백테스트 코인: {n}개")
    print(f"  메인 단독  평균 {df_res['main_profit_pct'].mean():+7.2f}%  "
          f"중앙값 {df_res['main_profit_pct'].median():+7.2f}%  "
          f"흑자 {main_pos}/{n} ({main_pos/n*100:.0f}%)")
    print(f"  CT 단독    평균 {df_res['ct_profit_pct'].mean():+7.2f}%  "
          f"중앙값 {df_res['ct_profit_pct'].median():+7.2f}%  "
          f"흑자 {ct_pos}/{n} ({ct_pos/n*100:.0f}%)")
    print(f"  합산(권장) 평균 {df_res['combined_profit_pct'].mean():+7.2f}%  "
          f"중앙값 {df_res['combined_profit_pct'].median():+7.2f}%  "
          f"흑자 {combined_pos}/{n} ({combined_pos/n*100:.0f}%)")
    print(f"  합산 평균 MDD: {df_res['combined_mdd'].mean():.1f}%  "
          f"중앙값 MDD: {df_res['combined_mdd'].median():.1f}%  "
          f"최대 MDD: {df_res['combined_mdd'].max():.1f}%")

    # CSV 저장
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    out_path = out_dir / f"newlistings_{args.tf}_{today}.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nCSV 저장: {out_path}")


if __name__ == "__main__":
    main()
