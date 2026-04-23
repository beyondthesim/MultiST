"""
Multi Supertrend 백테스트 프레임워크
사용법:
  python main.py                                  # params.json 기본값
  python main.py --symbol ETH/USDT:USDT --tf 15m
  python main.py --symbol BTC/USDT:USDT --start 2025-01-01 --end 2025-12-31
  python main.py --refresh                        # 캐시 무시하고 데이터 재수집
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Multi Supertrend Backtest")
    p.add_argument("--config",   default="config/params.json", help="파라미터 파일 경로")
    p.add_argument("--symbol",   help="거래 심볼 (예: ETH/USDT:USDT)")
    p.add_argument("--tf",       dest="timeframe", help="타임프레임 (예: 15m, 1h)")
    p.add_argument("--start",    dest="start_date", help="시작일 YYYY-MM-DD")
    p.add_argument("--end",      dest="end_date",   help="종료일 YYYY-MM-DD")
    p.add_argument("--refresh",  action="store_true", help="캐시 재수집")
    p.add_argument("--capital",  type=float, help="초기 자본 (USDT)")
    p.add_argument("--factor",   type=float, help="Supertrend factor 덮어쓰기")
    p.add_argument("--atr",      type=int,   help="Supertrend ATR period 덮어쓰기")
    return p.parse_args()


def load_params(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(params: dict, args) -> dict:
    """CLI 인자로 params 덮어쓰기"""
    if args.symbol:
        params["symbol"] = args.symbol
    if args.timeframe:
        params["timeframe"] = args.timeframe
    if args.start_date:
        params["start_date"] = args.start_date
    if args.end_date:
        params["end_date"] = args.end_date
    if args.capital:
        params["initial_capital"] = args.capital
    if args.factor:
        params["st1"]["factor"] = args.factor
        params["st2"]["factor"] = args.factor
    if args.atr:
        params["st1"]["atr_period"] = args.atr
        params["st2"]["atr_period"] = args.atr
    return params


def main():
    args   = parse_args()
    params = load_params(args.config)
    params = apply_overrides(params, args)

    print(f"\n{'='*60}")
    print(f"  Multi Supertrend Backtest")
    print(f"  심볼: {params['symbol']}  타임프레임: {params['timeframe']}")
    print(f"  기간: {params['start_date']} ~ {params['end_date']}")
    print(f"  ST factor={params['st1']['factor']}  ATR={params['st1']['atr_period']}")
    print(f"{'='*60}")

    # ── 데이터 수집 ──────────────────────────────────────────────────
    from data.fetcher import load_ohlcv_with_warmup
    print("\n[1] 데이터 수집")
    df = load_ohlcv_with_warmup(
        symbol       = params["symbol"],
        timeframe    = params["timeframe"],
        start_date   = params["start_date"],
        end_date     = params["end_date"],
        exchange_id  = params.get("exchange", "okx"),
        warmup_bars  = 200,
        force_refresh= args.refresh,
    )
    print(f"  전체 봉 수: {len(df):,}  (워밍업 포함)")
    print(f"  백테스트 구간: {df[df['in_backtest']].index[0].strftime('%Y-%m-%d')} "
          f"~ {df.index[-1].strftime('%Y-%m-%d')}")

    # ── 신호 생성 ─────────────────────────────────────────────────────
    from strategy.signal import build_signals
    print("\n[2] 지표 & 신호 계산")
    df = build_signals(df, params)

    # 워밍업 구간 제거 (백테스트 구간만 사용)
    df_bt = df[df["in_backtest"]].copy()
    print(f"  백테스트 봉 수: {len(df_bt):,}")
    long_sigs  = df_bt["long_entry"].sum()
    short_sigs = df_bt["short_entry"].sum()
    print(f"  롱 신호: {long_sigs}  숏 신호: {short_sigs}")

    # ── 백테스트 실행 ─────────────────────────────────────────────────
    from backtest.engine import BacktestEngine
    print("\n[3] 백테스트 실행")
    engine = BacktestEngine(params)
    main_trades, ct_trades, equity_curve = engine.run(df_bt)
    ct_enabled = params.get("counter_trend", {}).get("enabled", False)
    print(f"  완료: 메인 {len(main_trades)}건"
          + (f" | 역추세 {len(ct_trades)}건" if ct_enabled else ""))

    # ── 리포트 출력 ───────────────────────────────────────────────────
    from backtest.reporter import print_report
    print()
    print_report(
        main_trades, equity_curve, params,
        title="Supertrend 3/5 TP 백테스트",
        ct_trades=ct_trades if ct_enabled else [],
    )

    # 거래 내역 CSV 저장
    import pandas as pd
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    safe_sym = params["symbol"].replace("/", "_").replace(":", "_")

    if main_trades:
        out_path = out_dir / f"{safe_sym}_{params['timeframe']}_{params['start_date']}.csv"
        pd.DataFrame(main_trades).to_csv(out_path, index=False)
        print(f"\n  메인 거래 내역: {out_path}")

    if ct_trades:
        ct_path = out_dir / f"{safe_sym}_{params['timeframe']}_{params['start_date']}_ct.csv"
        pd.DataFrame(ct_trades).to_csv(ct_path, index=False)
        print(f"  역추세 거래 내역: {ct_path}")


if __name__ == "__main__":
    main()
