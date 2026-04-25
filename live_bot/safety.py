"""
안전 장치

체크 항목 (매 루프):
  1. KILL_SWITCH 파일 존재 → 즉시 모든 포지션 청산 + 봇 정지
  2. 일일 손실 한도 초과 → 당일 신규 진입 정지 (보유 포지션은 유지)
  3. MDD 한도 초과 → 봇 영구 정지 (수동 리셋 필요)
"""
from live_bot import config, notifier, state as state_mod
from live_bot.exchange import OKXClient


def check_kill_switch(client: OKXClient, symbol: str, state: dict) -> bool:
    """kill.txt 파일 존재 시 비상 청산 후 정지. True 반환 시 즉시 봇 종료해야 함."""
    if config.KILL_SWITCH_FILE.exists():
        notifier.alert(f"KILL SWITCH 감지 ({config.KILL_SWITCH_FILE}). 모든 포지션 청산 진행.")
        try:
            client.close_all_positions(symbol)
        except Exception as e:
            notifier.error(f"비상 청산 실패: {e}")
        state["halted"]      = True
        state["halt_reason"] = "kill_switch"
        state_mod.save(state)
        return True
    return False


def check_daily_loss(state: dict, current_equity: float) -> bool:
    """일일 -X% 도달 시 당일 신규 진입 차단. True 반환 시 신규 진입 금지."""
    pnl = state_mod.daily_pnl_pct(state, current_equity)
    if pnl <= -config.MAX_DAILY_LOSS_PCT:
        notifier.alert(f"일일 손실 한도 도달: {pnl:.2f}% ≤ -{config.MAX_DAILY_LOSS_PCT}%. 당일 신규 진입 차단.")
        return True
    return False


def check_max_dd(client: OKXClient, symbol: str, state: dict, current_equity: float) -> bool:
    """누적 MDD 한도 도달 시 봇 영구 정지. True 반환 시 즉시 봇 종료해야 함."""
    dd = state_mod.current_dd_pct(state, current_equity)
    if dd <= -config.MAX_DD_PCT:
        notifier.alert(
            f"MDD 한도 도달: {dd:.2f}% ≤ -{config.MAX_DD_PCT}%. 모든 포지션 청산 후 봇 영구 정지."
        )
        try:
            client.close_all_positions(symbol)
        except Exception as e:
            notifier.error(f"청산 실패: {e}")
        state["halted"]      = True
        state["halt_reason"] = f"max_dd ({dd:.2f}%)"
        state_mod.save(state)
        return True
    return False


def is_halted(state: dict) -> bool:
    return bool(state.get("halted"))
