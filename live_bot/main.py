"""
MultiST 라이브 트레이딩 봇 메인 루프

실행:
  # DRY-RUN (기본, 주문 안 함)
  python -m live_bot.main

  # 실거래 (명시적 환경변수)
  LIVE_MODE=true python -m live_bot.main

플로우:
  1. 안전 체크 (kill switch, halt 상태)
  2. 자본 갱신 + MDD 체크
  3. 새 봉 도착 대기/확인
  4. 신호 계산
  5. 포지션 처리:
     a) 메인 SL/TP 체크 → 청산 시 새 진입 가능
     b) 메인 ST 역전 청산 (close_long/short)
     c) CT DCA 체크
     d) CT SL/TP 체크
     e) CT ST 회복 청산 (옵션)
     f) 메인 신규 진입 (signal & 포지션 없음)
     g) CT 신규 진입
  6. 거래소 동기화 체크 (5분마다)
"""
import sys
import time
from datetime import datetime, timezone

from live_bot import config, notifier, state as state_mod
from live_bot.exchange import OKXClient
from live_bot.signal_runner import compute_latest_signals
from live_bot.position_manager import fetch_total_equity, reconcile_positions
from live_bot import safety, order_executor as oe


def initialize(client: OKXClient, state: dict) -> float:
    """첫 실행 시 자본/상태 초기화"""
    eq = fetch_total_equity(client)
    if eq <= 0:
        notifier.error("거래소 잔고 0 또는 조회 실패. .env.secret API 키 확인.")
        sys.exit(1)

    if state.get("initial_capital") is None:
        state["initial_capital"]    = eq
        state["peak_equity"]        = eq
        state["daily_start_equity"] = eq
        state["daily_date"]         = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state_mod.save(state)
        notifier.info(f"첫 실행. 초기 자본 기록: ${eq:,.2f}")
    return eq


def process_signals(
    client: OKXClient, params: dict, state: dict,
    signal: dict, eq: float, daily_locked: bool,
) -> None:
    """단일 봉 신호 처리"""
    symbol = params["symbol"]

    # ── (a) 메인 SL/TP 체크 ────────────────────────────────────────
    if state.get("main"):
        if oe.check_main_sl_tp(client, symbol, params, state, signal):
            pass  # 청산됨, 다음 봉부터 새 진입 가능

    # ── (b) 메인 ST 역전 청산 ─────────────────────────────────────
    if state.get("main"):
        pos = state["main"]
        if pos["side"] == "long" and signal["close_long"]:
            oe.close_main(client, symbol, state, reason="ST_FLIP_long")
        elif pos["side"] == "short" and signal["close_short"]:
            oe.close_main(client, symbol, state, reason="ST_FLIP_short")

    # ── (c) CT DCA 체크 ────────────────────────────────────────────
    if state.get("ct"):
        oe.check_ct_dca(client, symbol, params, state, signal)

    # ── (d) CT SL/TP 체크 ─────────────────────────────────────────
    if state.get("ct"):
        oe.check_ct_sl_tp(client, symbol, params, state, signal)

    # ── (e) CT ST 회복 청산 (ct_exit_enabled=False면 무시) ────────
    ct_cfg = params.get("counter_trend", {})
    if state.get("ct") and ct_cfg.get("ct_exit_enabled", False):
        ctp = state["ct"]
        if ctp["side"] == "long" and signal["ct_close_long"]:
            oe.close_ct(client, symbol, state, reason="CT_EXIT_long")
        elif ctp["side"] == "short" and signal["ct_close_short"]:
            oe.close_ct(client, symbol, state, reason="CT_EXIT_short")

    # ── (f) 메인 신규 진입 ────────────────────────────────────────
    if not state.get("main") and not daily_locked:
        if signal["long_entry"] or signal["short_entry"]:
            oe.open_main(client, symbol, params, state, signal, eq)

    # ── (g) CT 신규 진입 ──────────────────────────────────────────
    if not state.get("ct") and not daily_locked and ct_cfg.get("enabled", False):
        if signal["ct_long_entry"]:
            sig_long = dict(signal); sig_long["ct_long_entry"] = True; sig_long["ct_short_entry"] = False
            oe.open_ct(client, symbol, params, state, sig_long, eq)
        elif signal["ct_short_entry"]:
            sig_short = dict(signal); sig_short["ct_short_entry"] = True; sig_short["ct_long_entry"] = False
            oe.open_ct(client, symbol, params, state, sig_short, eq)


def main():
    print(config.summary())

    client = OKXClient()
    if not client.ping():
        notifier.error("OKX 연결 실패. 종료.")
        sys.exit(1)
    notifier.info(f"OKX 연결 성공. 모드: {'LIVE' if config.LIVE_MODE else 'DRY-RUN'}")

    params = config.load_params()
    notifier.info(f"심볼: {params['symbol']} / TF: {params['timeframe']}")

    state = state_mod.load()
    if safety.is_halted(state):
        notifier.error(f"봇이 정지 상태 ({state.get('halt_reason')}). state 파일 확인 후 수정.")
        sys.exit(1)

    eq = initialize(client, state)
    last_sync_ts = 0

    while True:
        try:
            # ── 안전 체크 ────────────────────────────────────────
            if safety.check_kill_switch(client, params["symbol"], state):
                break

            eq = fetch_total_equity(client)
            new_day = state_mod.reset_daily_if_needed(state, eq)
            if new_day:
                notifier.info(f"새로운 UTC 일자. 일일 시작 자본 갱신: ${eq:,.2f}")
            state_mod.update_peak(state, eq)

            if safety.check_max_dd(client, params["symbol"], state, eq):
                break

            daily_locked = safety.check_daily_loss(state, eq)

            # ── 신호 계산 ────────────────────────────────────────
            signal = compute_latest_signals(client, params)
            if not signal:
                notifier.warn("신호 계산 실패. 30초 후 재시도.")
                time.sleep(30)
                continue

            # 새 봉 확정 여부
            current_ts = str(signal["ts"])
            if state.get("last_bar_ts") != current_ts:
                notifier.info(
                    f"새 봉 도착: {current_ts} close=${signal['close']:.4f} | "
                    f"L={signal['long_entry']} S={signal['short_entry']} "
                    f"CL={signal['ct_long_entry']} CS={signal['ct_short_entry']} | "
                    f"BTCbull={signal['btc_bull']} BTCbear={signal['btc_bear']} | "
                    f"Eq=${eq:,.2f} DD={state_mod.current_dd_pct(state, eq):.2f}%"
                )
                process_signals(client, params, state, signal, eq, daily_locked)
                state["last_bar_ts"] = current_ts
                state_mod.save(state)

            # ── 포지션 동기화 (5분마다) ──────────────────────────
            now_ts = time.time()
            if now_ts - last_sync_ts >= config.SYNC_POSITIONS_SEC:
                reconcile_positions(client, params["symbol"], state)
                last_sync_ts = now_ts

            time.sleep(config.LOOP_INTERVAL_SEC)

        except KeyboardInterrupt:
            notifier.info("Ctrl+C 감지. 봇 종료 (포지션은 유지).")
            break
        except Exception as e:
            notifier.error(f"루프 예외: {e}")
            import traceback
            notifier.error(traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    main()
