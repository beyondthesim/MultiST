"""
실시간 봉 데이터 수신

OKX는 12m을 직접 지원하지 않으므로 3m을 받아 12m으로 리샘플링.
12m 봉 close 시점에 새 봉 도착 → 신호 계산 트리거.
"""
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from live_bot import config, notifier
from live_bot.exchange import OKXClient


_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
_RESAMPLE = {
    "12m": ("3m", "12min"),
    "2m":  ("1m", "2min"),
    "6m":  ("3m", "6min"),
}


def fetch_resampled(client: OKXClient, symbol: str, timeframe: str, bars_needed: int) -> pd.DataFrame:
    """
    필요한 봉수만큼 데이터를 받아 리샘플링.
    OKX는 한 번에 최대 300봉 → since 페이지네이션으로 수집.
    """
    if timeframe in _RESAMPLE:
        raw_tf, alias = _RESAMPLE[timeframe]
        target_min = int(timeframe.replace("m", ""))
        raw_min    = _TF_MIN[raw_tf]
        raw_needed = bars_needed * (target_min // raw_min) + 50
    elif timeframe.endswith("h"):
        raw_tf  = timeframe
        alias   = None
        raw_needed = bars_needed + 10
        raw_min = _TF_MIN[raw_tf]
    else:
        raw_tf  = timeframe
        alias   = None
        raw_needed = bars_needed + 10
        raw_min = _TF_MIN[raw_tf]

    # since 페이지네이션
    now_ms     = int(time.time() * 1000)
    span_ms    = raw_needed * raw_min * 60 * 1000 + 60 * 60 * 1000  # 여유 1시간
    since_ms   = now_ms - span_ms

    all_bars = []
    current  = since_ms
    while current < now_ms:
        try:
            bars = client.ex.fetch_ohlcv(symbol, raw_tf, since=current, limit=300)
        except Exception as e:
            notifier.error(f"OHLCV 페치 실패 ({symbol} {raw_tf}): {e}")
            break
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
        if len(bars) < 300:
            break

    if not all_bars:
        return pd.DataFrame()

    # 중복 제거 + 정렬
    seen, dedup = set(), []
    for b in all_bars:
        if b[0] not in seen:
            seen.add(b[0])
            dedup.append(b)
    all_bars = sorted(dedup, key=lambda x: x[0])

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # 미완성 마지막 봉 제거 (현재 시각이 봉 종료 전이면)
    last_ts = df.index[-1]
    bar_close = last_ts + pd.Timedelta(minutes=_TF_MIN.get(raw_tf, 1))
    if datetime.now(timezone.utc) < bar_close:
        df = df.iloc[:-1]

    if alias:
        df = (
            df.resample(alias, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
        )

    return df.tail(bars_needed)


def is_new_bar_closed(prev_last_ts: Optional[pd.Timestamp], current_last_ts: pd.Timestamp) -> bool:
    """이전에 본 마지막 봉 시각과 현재 마지막 봉 시각이 다르면 새 봉 확정"""
    if prev_last_ts is None:
        return True
    return current_last_ts > prev_last_ts


def wait_until_next_bar(timeframe: str, buffer_sec: int = 5) -> None:
    """다음 봉 종료 시각까지 대기 (12m 봉이면 12분 단위)"""
    tf_min = int(timeframe.replace("m", "").replace("h", "")) * (60 if "h" in timeframe else 1)
    now = datetime.now(timezone.utc)
    minutes_into = (now.minute % tf_min) * 60 + now.second
    wait = (tf_min * 60 - minutes_into) + buffer_sec
    notifier.info(f"다음 {timeframe} 봉까지 {wait}초 대기...")
    time.sleep(max(1, wait))
