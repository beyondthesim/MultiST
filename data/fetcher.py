"""
OHLCV 데이터 수집 모듈
- OKX API (ccxt) 사용
- 12m 봉 미지원 시 1m 봉 수집 후 리샘플링
- 로컬 parquet 캐싱으로 반복 실행 최적화
"""

import os
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# OKX가 직접 지원하는 타임프레임 목록
OKX_SUPPORTED_TF = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "12h",
    "1d", "2d", "3d", "1w", "1M",
}

# 리샘플링 규칙: (소스 타임프레임, pandas resample alias)
# 가능한 한 큰 소스 TF로 API 요청 수 최소화
RESAMPLE_MAP = {
    "12m": ("3m",  "12min"),   # 3m × 4 = 12m  (1m 대비 요청수 4분의 1)
    "2m":  ("1m",  "2min"),
    "6m":  ("3m",  "6min"),
    "10m": ("5m",  "10min"),
    "20m": ("5m",  "20min"),
    "45m": ("15m", "45min"),
    "2h":  ("1h",  "2h"),
    "3h":  ("1h",  "3h"),
}


def _load_exchange(exchange_id: str) -> ccxt.Exchange:
    """
    OHLCV 데이터 수집용 거래소 객체 생성
    공개 API 사용 (인증 불필요) - 시세 데이터는 인증 없이 접근 가능
    실제 거래용 API 키는 .env.secret에 보관하나 여기서는 미사용
    """
    ExchangeClass = getattr(ccxt, exchange_id)
    ex = ExchangeClass({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    return ex


def _cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{exchange_id}_{safe_symbol}_{timeframe}.parquet"


def _fetch_all_ohlcv(ex: ccxt.Exchange, symbol: str, tf: str,
                     since_ms: int, until_ms: int) -> pd.DataFrame:
    """API를 페이지네이션하여 전체 기간 수집"""
    all_bars = []
    limit = 300
    current = since_ms

    print(f"  [{symbol} {tf}] 데이터 수집 중...", end="", flush=True)
    batch = 0
    while current < until_ms:
        try:
            bars = ex.fetch_ohlcv(symbol, tf, since=current, limit=limit)
        except Exception as e:
            print(f"\n  API 오류: {e}. 10초 후 재시도...")
            time.sleep(10)
            continue

        if not bars:
            break

        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1

        batch += 1
        if batch % 20 == 0:
            print(".", end="", flush=True)
        time.sleep(ex.rateLimit / 1000)

    print(f" 완료 ({len(all_bars):,}개)")
    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    df = df[~df.index.duplicated(keep="first")]
    return df


def _resample_to_tf(df: pd.DataFrame, target_alias: str) -> pd.DataFrame:
    """1m 데이터를 타겟 타임프레임으로 리샘플링"""
    ohlcv = df.resample(target_alias).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return ohlcv


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    exchange_id: str = "okx",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    OHLCV 데이터 로드 (캐시 우선, 없으면 API 수집)

    Parameters
    ----------
    symbol      : 'PI/USDT:USDT' 형식 (ccxt 표준)
    timeframe   : '12m', '1h' 등
    start_date  : 'YYYY-MM-DD'
    end_date    : 'YYYY-MM-DD'
    exchange_id : 'okx'
    force_refresh: True이면 캐시 무시하고 재수집
    """
    cache_path = _cache_path(exchange_id, symbol, timeframe)
    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end_date,   tz="UTC").timestamp() * 1000)

    # 캐시 확인
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index, utc=True)
        cached_start = df.index[0]
        cached_end   = df.index[-1]
        need_start   = pd.Timestamp(start_date, tz="UTC")
        need_end     = pd.Timestamp(end_date,   tz="UTC")

        if cached_start <= need_start and cached_end >= need_end - pd.Timedelta(timeframe_to_minutes(timeframe), "min") * 2:
            print(f"  캐시 사용: {cache_path.name}")
            mask = (df.index >= need_start) & (df.index < need_end)
            return df[mask].copy()
        print(f"  캐시 범위 부족 → 재수집")

    ex = _load_exchange(exchange_id)

    tf_lower = timeframe.lower()
    if tf_lower in OKX_SUPPORTED_TF:
        # 직접 지원하는 타임프레임
        raw_tf    = tf_lower
        resample  = None
    elif timeframe in RESAMPLE_MAP:
        raw_tf, resample = RESAMPLE_MAP[timeframe]
        print(f"  {timeframe}은 OKX 미지원 → {raw_tf} 수집 후 리샘플링")
    else:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}. RESAMPLE_MAP에 추가하세요.")

    # 여유분 추가 (지표 워밍업용 - 최대 200봉)
    warmup_minutes = 200 * timeframe_to_minutes(timeframe)
    fetch_since_ms = since_ms - warmup_minutes * 60 * 1000

    df_raw = _fetch_all_ohlcv(ex, symbol, raw_tf, fetch_since_ms, until_ms)

    if df_raw.empty:
        raise RuntimeError(f"데이터 없음: {symbol} {raw_tf}")

    if resample:
        df_raw = _resample_to_tf(df_raw, resample)

    df_raw.to_parquet(cache_path)
    print(f"  캐시 저장: {cache_path.name}")

    mask = (df_raw.index >= pd.Timestamp(start_date, tz="UTC")) & \
           (df_raw.index <  pd.Timestamp(end_date,   tz="UTC"))
    return df_raw[mask].copy()


def load_ohlcv_with_warmup(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    exchange_id: str = "okx",
    warmup_bars: int = 200,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    지표 계산용 워밍업 포함 전체 데이터 반환
    반환 DataFrame에 'in_backtest' 컬럼 추가 (실제 백테스트 구간 마킹)
    """
    cache_path = _cache_path(exchange_id, symbol, timeframe)
    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end_date,   tz="UTC").timestamp() * 1000)

    warmup_minutes = warmup_bars * timeframe_to_minutes(timeframe)
    fetch_since_ms = since_ms - warmup_minutes * 60 * 1000

    # 캐시 확인
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index, utc=True)
        cached_start = df.index[0]
        cached_end   = df.index[-1]
        need_start   = pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(warmup_minutes, "min")
        need_end     = pd.Timestamp(end_date,   tz="UTC")

        if cached_start <= need_start and cached_end >= need_end - pd.Timedelta(timeframe_to_minutes(timeframe), "min") * 2:
            print(f"  캐시 사용: {cache_path.name}")
            mask = (df.index >= pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(warmup_minutes, "min")) & \
                   (df.index <  need_end)
            df = df[mask].copy()
            df["in_backtest"] = df.index >= pd.Timestamp(start_date, tz="UTC")
            return df

    ex = _load_exchange(exchange_id)

    tf_lower = timeframe.lower()
    if tf_lower in OKX_SUPPORTED_TF:
        raw_tf   = tf_lower
        resample = None
    elif timeframe in RESAMPLE_MAP:
        raw_tf, resample = RESAMPLE_MAP[timeframe]
        print(f"  {timeframe}은 OKX 미지원 → {raw_tf} 수집 후 리샘플링")
    else:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

    df_raw = _fetch_all_ohlcv(ex, symbol, raw_tf, fetch_since_ms, until_ms)

    if df_raw.empty:
        raise RuntimeError(f"데이터 없음: {symbol} {raw_tf}")

    if resample:
        df_raw = _resample_to_tf(df_raw, resample)

    df_raw.to_parquet(cache_path)
    print(f"  캐시 저장: {cache_path.name}")

    mask = (df_raw.index >= pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(warmup_minutes, "min")) & \
           (df_raw.index <  pd.Timestamp(end_date,   tz="UTC"))
    df_out = df_raw[mask].copy()
    df_out["in_backtest"] = df_out.index >= pd.Timestamp(start_date, tz="UTC")
    return df_out


def timeframe_to_minutes(tf: str) -> int:
    """타임프레임 문자열 → 분 단위 정수 변환"""
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    if tf.endswith("w"):
        return int(tf[:-1]) * 10080
    raise ValueError(f"알 수 없는 타임프레임 형식: {tf}")
