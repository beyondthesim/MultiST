"""
포지션 매니저

핵심:
  - 봇 내부 상태(state.json)와 거래소 실제 포지션 동기화
  - 메인 + CT 양방향(hedge) 슬롯 추적
  - DCA 추적 (CT는 평균단가, 진입 횟수)
  - SL/TP 가격 계산

거래소가 양방향 모드라 long/short 슬롯이 별도. 봇은:
  - long  슬롯 = 메인 LONG 또는 CT LONG (동시 1개만 가능 - 메인이 우선)
  - short 슬롯 = 메인 SHORT 또는 CT SHORT (동시 1개만 가능)

엔진 백테스트와 동일하게 메인과 CT는 슬롯 분리 운용:
  하지만 양방향 모드 1슬롯 제한으로 같은 방향 메인+CT는 합쳐짐
  → 단순화를 위해 봇 내부에서 메인/CT를 logical 분리 추적
"""
from typing import Optional

from live_bot import notifier
from live_bot.exchange import OKXClient


def calc_size_in_base(notional_usdt: float, price: float, contract_size: float = 1.0) -> float:
    """
    USDT 명목을 base 코인 수량(=계약 수)으로 변환.
    OKX PI/USDT:USDT 영구는 보통 contractSize=1 (1계약 = 1 PI).
    """
    return round(notional_usdt / (price * contract_size), 4)


def fetch_total_equity(client: OKXClient) -> float:
    """현재 총 자본 (USDT 잔고)"""
    return client.fetch_balance_usdt()


def reconcile_positions(client: OKXClient, symbol: str, state: dict) -> None:
    """
    거래소 실제 포지션과 봇 상태 비교.
    불일치 시 경고 (자동 수정은 위험하므로 사람이 개입 필요).
    """
    real = client.fetch_positions(symbol)
    real_long  = real["long"]  is not None
    real_short = real["short"] is not None

    bot_long_active  = bool(
        (state.get("main") and state["main"]["side"] == "long") or
        (state.get("ct")   and state["ct"]["side"]   == "long")
    )
    bot_short_active = bool(
        (state.get("main") and state["main"]["side"] == "short") or
        (state.get("ct")   and state["ct"]["side"]   == "short")
    )

    if real_long != bot_long_active:
        notifier.warn(
            f"포지션 불일치 LONG: 거래소={real_long} vs 봇={bot_long_active}"
        )
    if real_short != bot_short_active:
        notifier.warn(
            f"포지션 불일치 SHORT: 거래소={real_short} vs 봇={bot_short_active}"
        )
