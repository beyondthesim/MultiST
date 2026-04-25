"""
봇 상태 영속화 — 재시작 시 진행 중 포지션/통계 복원

저장 항목:
  - 시작 자본 (초기 1회)
  - 일일 시작 자본 (매일 00:00 UTC 갱신)
  - 일일 손익
  - 누적 MDD 추적용 peak equity
  - 마지막 처리한 봉 timestamp (중복 신호 방지)
  - 메인/CT 진입 가격 + 평균단가 (DCA 추적)
  - 메인/CT 진입 시점의 봇 측 entry_equity (PnL 계산용)
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from live_bot import config


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


_DEFAULT = {
    "initial_capital":   None,
    "peak_equity":       None,
    "daily_start_equity": None,
    "daily_date":        None,
    "last_bar_ts":       None,
    # 메인 포지션 상태 (None = 없음)
    "main": None,        # {"side": "long"|"short", "entry_price", "entry_size", "ts"}
    # CT 포지션 상태 (DCA 추적)
    "ct": None,          # {"side", "avg_price", "total_size", "dca_count", "last_entry_price", "ts"}
    "halted":            False,
    "halt_reason":       "",
}


def load() -> dict:
    if not config.STATE_FILE.exists():
        return dict(_DEFAULT)
    with open(config.STATE_FILE, encoding="utf-8") as f:
        s = json.load(f)
    # 누락 키 보강
    for k, v in _DEFAULT.items():
        s.setdefault(k, v)
    return s


def save(state: dict) -> None:
    state["_updated_at"] = _now_utc()
    with open(config.STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def reset_daily_if_needed(state: dict, current_equity: float) -> bool:
    """매일 UTC 00:00에 일일 시작 자본 갱신. 갱신 시 True."""
    today = _today_utc()
    if state.get("daily_date") != today:
        state["daily_date"]         = today
        state["daily_start_equity"] = current_equity
        return True
    return False


def update_peak(state: dict, current_equity: float) -> None:
    """MDD 추적용 최고점 갱신"""
    peak = state.get("peak_equity") or current_equity
    state["peak_equity"] = max(peak, current_equity)


def daily_pnl_pct(state: dict, current_equity: float) -> float:
    start = state.get("daily_start_equity") or current_equity
    if start <= 0:
        return 0.0
    return (current_equity - start) / start * 100


def current_dd_pct(state: dict, current_equity: float) -> float:
    peak = state.get("peak_equity") or current_equity
    if peak <= 0:
        return 0.0
    return (current_equity - peak) / peak * 100  # 음수
