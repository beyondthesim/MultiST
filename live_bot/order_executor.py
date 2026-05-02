"""
주문 실행 로직

설계 단순화:
  - 12분봉 close 시점에 모든 결정 (SL/TP/DCA/진입/청산)
  - 봉 close 가격으로 SL/TP 판정 → 시장가 주문
  - DCA 평균단가는 봇 내부에서 추적
  - 거래소는 양방향 모드 + 크로스 마진 가정

지원 동작:
  메인:
    - 신규 진입 (long_entry / short_entry)
    - 청산 (close_long / close_short - ST 역전)
    - SL 청산 (long_sl_pct / short_sl_pct)
    - TP 청산 (long_tp / short_tp 다단계)

  CT:
    - 신규 진입 (ct_long_entry / ct_short_entry)
    - DCA 추가 (가격 1.5% 추가 역행 시)
    - SL 청산 (sl_long_pct / sl_short_pct)
    - TP 청산 (ct_long_tp / ct_short_tp - 평균단가 기준)
    - CT_EXIT (ct_close_long / ct_close_short - 옵션)
"""
import time
from typing import Optional

from live_bot import notifier, trade_logger
from live_bot.exchange import OKXClient
from live_bot.position_manager import calc_size_in_base


# ── 메인 포지션 ────────────────────────────────────────────────────────
def open_main(
    client: OKXClient, symbol: str, params: dict, state: dict,
    signal: dict, total_equity: float,
) -> None:
    """메인 신규 진입"""
    main_pct = params.get("main_position_pct", 200)
    notional = total_equity * main_pct / 100.0
    price    = signal["close"]
    qty      = calc_size_in_base(notional, price)
    if qty <= 0:
        notifier.warn(f"메인 진입 수량 계산 실패: notional={notional}, price={price}")
        return

    is_long  = signal["long_entry"]
    side_dir = "long" if is_long else "short"
    side_op  = "buy"  if is_long else "sell"
    sl_pct   = params["long_sl_pct"] if is_long else params["short_sl_pct"]

    notifier.trade(f"메인 {side_dir.upper()} 진입 시도: qty={qty} @ ${price:.4f} "
                   f"(명목 ${notional:,.0f}, 자본 ${total_equity:,.0f})")
    client.create_market_order(symbol, side_op, qty, pos_side=side_dir,
                                client_order_id=f"main_{int(time.time())}")
    trade_logger.log_event(
        strategy="main", symbol=symbol, action="entry", direction=side_dir,
        price=price, qty=qty, notional=notional,
        entry_seed=state.get("initial_capital"),
        entry_total_equity=total_equity,
    )

    state["main"] = {
        "side":         side_dir,
        "entry_price":  price,
        "entry_size":   qty,
        "entry_notional": notional,
        "ts":           str(signal["ts"]),
        "sl_price":     price * ((1 - sl_pct) if is_long else (1 + sl_pct)),
        "tp_done":      [],   # 이미 체결한 TP 단계 인덱스
    }


def close_main(client: OKXClient, symbol: str, state: dict, reason: str,
                exit_price: Optional[float] = None) -> None:
    """메인 전량 청산"""
    if not state.get("main"):
        return
    pos = state["main"]
    side_op = "sell" if pos["side"] == "long" else "buy"
    notifier.trade(f"메인 청산 ({reason}): {pos['side'].upper()} qty={pos['entry_size']}")
    client.create_market_order(symbol, side_op, pos["entry_size"], pos_side=pos["side"],
                                reduce_only=True, client_order_id=f"main_close_{int(time.time())}")
    if exit_price is not None:
        pnl = trade_logger.pnl_for(pos["side"], pos["entry_price"], exit_price, pos["entry_size"])
        seed = state.get("initial_capital")
        pnl_pct = (pnl / seed * 100) if seed else None
        trade_logger.log_event(
            strategy="main", symbol=symbol, action="close", direction=pos["side"],
            price=exit_price, qty=pos["entry_size"],
            notional=pos.get("entry_notional", 0),
            pnl=pnl, reason=reason,
            entry_seed=seed, pnl_pct=pnl_pct,
        )
    state["main"] = None


def check_main_sl_tp(
    client: OKXClient, symbol: str, params: dict, state: dict,
    signal: dict,
) -> bool:
    """메인 SL/TP 체크. 청산 발생 시 True."""
    if not state.get("main"):
        return False
    pos = state["main"]
    price = signal["close"]
    is_long = pos["side"] == "long"

    # SL 체크
    if (is_long and price <= pos["sl_price"]) or (not is_long and price >= pos["sl_price"]):
        close_main(client, symbol, state, reason=f"SL @${pos['sl_price']:.4f}", exit_price=price)
        return True

    # TP 단계 체크 (다단계 부분 청산)
    tp_list = params["long_tp"] if is_long else params["short_tp"]
    entry_price = pos["entry_price"]
    for i, tp in enumerate(tp_list):
        if i in pos["tp_done"]:
            continue
        tp_price = entry_price * ((1 + tp["pct"]) if is_long else (1 - tp["pct"]))
        hit = (is_long and price >= tp_price) or (not is_long and price <= tp_price)
        if hit:
            qty_close = pos["entry_size"] * (tp["qty_pct"] / 100.0)
            side_op   = "sell" if is_long else "buy"
            notifier.trade(f"메인 TP{i+1} ({tp['pct']*100:.1f}%): qty={qty_close:.4f} @${tp_price:.4f}")
            client.create_market_order(symbol, side_op, qty_close, pos_side=pos["side"],
                                        reduce_only=True,
                                        client_order_id=f"main_tp{i}_{int(time.time())}")
            pos["tp_done"].append(i)
            partial_pnl = trade_logger.pnl_for(pos["side"], entry_price, tp_price, qty_close)
            full_close = tp["qty_pct"] >= 100
            seed = state.get("initial_capital")
            partial_pct = (partial_pnl / seed * 100) if seed else None
            trade_logger.log_event(
                strategy="main", symbol=symbol,
                action="close" if full_close else "tp_partial",
                direction=pos["side"],
                price=tp_price, qty=qty_close,
                notional=qty_close * tp_price,
                pnl=partial_pnl, reason=f"TP{i+1}",
                entry_seed=seed, pnl_pct=partial_pct,
            )
            if full_close:
                state["main"] = None
                return True
    return False


# ── CT 포지션 ─────────────────────────────────────────────────────────
def open_ct(
    client: OKXClient, symbol: str, params: dict, state: dict,
    signal: dict, total_equity: float,
) -> None:
    """CT 신규 진입 (DCA 0회 상태)"""
    ct_cfg     = params["counter_trend"]
    ct_pos_pct = ct_cfg.get("ct_position_pct", 600)
    max_dca    = ct_cfg.get("max_dca", 0)
    weights    = ct_cfg.get("dca_weights", [])
    total_w    = 1 + sum(weights[:max_dca])

    # CT 풀 사이클 명목
    full_notional = total_equity * ct_pos_pct / 100.0
    per_unit_notional = full_notional / total_w  # 첫 진입 명목

    price = signal["close"]
    qty   = calc_size_in_base(per_unit_notional, price)
    if qty <= 0:
        notifier.warn(f"CT 진입 수량 계산 실패")
        return

    is_long  = signal["ct_long_entry"]
    side_dir = "long" if is_long else "short"
    side_op  = "buy"  if is_long else "sell"
    sl_pct   = ct_cfg["sl_long_pct"] if is_long else ct_cfg["sl_short_pct"]

    notifier.trade(f"CT {side_dir.upper()} 진입: qty={qty} @${price:.4f} "
                   f"(첫진입 ${per_unit_notional:.0f} / 풀 ${full_notional:,.0f})")
    client.create_market_order(symbol, side_op, qty, pos_side=side_dir,
                                client_order_id=f"ct_{int(time.time())}")
    trade_logger.log_event(
        strategy="ct", symbol=symbol, action="entry", direction=side_dir,
        price=price, qty=qty, notional=per_unit_notional, dca_level=0,
        entry_seed=state.get("initial_capital"),
        entry_total_equity=total_equity,
    )

    state["ct"] = {
        "side":              side_dir,
        "avg_price":         price,
        "total_size":        qty,
        "dca_count":         0,
        "last_entry_price":  price,
        "per_unit_notional": per_unit_notional,
        "ts":                str(signal["ts"]),
        "sl_pct":            sl_pct,
        "tp_done":           [],
    }


def close_ct(client: OKXClient, symbol: str, state: dict, reason: str,
              exit_price: Optional[float] = None) -> None:
    """CT 전량 청산"""
    if not state.get("ct"):
        return
    pos = state["ct"]
    side_op = "sell" if pos["side"] == "long" else "buy"
    notifier.trade(f"CT 청산 ({reason}): {pos['side'].upper()} qty={pos['total_size']:.4f} "
                   f"avg=${pos['avg_price']:.4f} dca={pos['dca_count']}")
    client.create_market_order(symbol, side_op, pos["total_size"], pos_side=pos["side"],
                                reduce_only=True, client_order_id=f"ct_close_{int(time.time())}")
    if exit_price is not None:
        pnl = trade_logger.pnl_for(pos["side"], pos["avg_price"], exit_price, pos["total_size"])
        seed = state.get("initial_capital")
        pnl_pct = (pnl / seed * 100) if seed else None
        trade_logger.log_event(
            strategy="ct", symbol=symbol, action="close", direction=pos["side"],
            price=exit_price, qty=pos["total_size"],
            notional=pos["total_size"] * exit_price,
            pnl=pnl, reason=reason, dca_level=pos.get("dca_count", 0),
            entry_seed=seed, pnl_pct=pnl_pct,
        )
    state["ct"] = None


def check_ct_dca(client: OKXClient, symbol: str, params: dict, state: dict,
                  signal: dict) -> None:
    """CT DCA 체크: 직전 진입가 대비 1.5% 추가 역행 시 추가 매수"""
    if not state.get("ct"):
        return
    pos     = state["ct"]
    ct_cfg  = params["counter_trend"]
    max_dca = ct_cfg.get("max_dca", 0)
    if pos["dca_count"] >= max_dca:
        return

    weights      = ct_cfg.get("dca_weights", [])
    dca_price_pct = ct_cfg.get("dca_price_pct", 0.015)
    price        = signal["close"]
    last_p       = pos["last_entry_price"]
    is_long      = pos["side"] == "long"

    trigger = (is_long and price < last_p * (1 - dca_price_pct)) or \
              (not is_long and price > last_p * (1 + dca_price_pct))

    if not trigger:
        return

    next_idx = pos["dca_count"]
    weight   = weights[next_idx] if next_idx < len(weights) else 1
    add_notional = pos["per_unit_notional"] * weight
    add_qty      = calc_size_in_base(add_notional, price)
    if add_qty <= 0:
        return

    side_op = "buy" if is_long else "sell"
    notifier.trade(f"CT DCA #{next_idx+1} (weight={weight}): qty={add_qty} @${price:.4f}")
    client.create_market_order(symbol, side_op, add_qty, pos_side=pos["side"],
                                client_order_id=f"ct_dca{next_idx}_{int(time.time())}")
    trade_logger.log_event(
        strategy="ct", symbol=symbol, action="dca", direction=pos["side"],
        price=price, qty=add_qty, notional=add_notional,
        dca_level=next_idx + 1,
        entry_seed=state.get("initial_capital"),
    )

    # 평균단가 갱신
    new_total = pos["total_size"] + add_qty
    new_avg   = (pos["avg_price"] * pos["total_size"] + price * add_qty) / new_total
    pos["avg_price"]        = new_avg
    pos["total_size"]       = new_total
    pos["last_entry_price"] = price
    pos["dca_count"]       += 1
    pos["tp_done"]          = []  # TP 재계산되므로 리셋


def check_ct_sl_tp(client: OKXClient, symbol: str, params: dict, state: dict,
                    signal: dict) -> bool:
    """CT SL/TP 체크 (평균단가 기준). 청산 발생 시 True."""
    if not state.get("ct"):
        return False
    pos     = state["ct"]
    ct_cfg  = params["counter_trend"]
    price   = signal["close"]
    is_long = pos["side"] == "long"
    avg     = pos["avg_price"]

    # SL (평균단가 기준)
    sl_price = avg * ((1 - pos["sl_pct"]) if is_long else (1 + pos["sl_pct"]))
    if (is_long and price <= sl_price) or (not is_long and price >= sl_price):
        close_ct(client, symbol, state, reason=f"SL @${sl_price:.4f} (avg ${avg:.4f})", exit_price=price)
        return True

    # TP 다단계 (평균단가 기준)
    tp_list = ct_cfg["ct_long_tp"] if is_long else ct_cfg["ct_short_tp"]
    for i, tp in enumerate(tp_list):
        if i in pos["tp_done"]:
            continue
        tp_price = avg * ((1 + tp["pct"]) if is_long else (1 - tp["pct"]))
        hit = (is_long and price >= tp_price) or (not is_long and price <= tp_price)
        if hit:
            qty_close = pos["total_size"] * (tp["qty_pct"] / 100.0)
            side_op   = "sell" if is_long else "buy"
            notifier.trade(f"CT TP{i+1} ({tp['pct']*100:.1f}%): qty={qty_close:.4f} @${tp_price:.4f}")
            client.create_market_order(symbol, side_op, qty_close, pos_side=pos["side"],
                                        reduce_only=True,
                                        client_order_id=f"ct_tp{i}_{int(time.time())}")
            partial_pnl = trade_logger.pnl_for(pos["side"], avg, tp_price, qty_close)
            full_close = tp["qty_pct"] >= 100 or (pos["total_size"] - qty_close) <= 1e-6
            seed = state.get("initial_capital")
            partial_pct = (partial_pnl / seed * 100) if seed else None
            trade_logger.log_event(
                strategy="ct", symbol=symbol,
                action="close" if full_close else "tp_partial",
                direction=pos["side"],
                price=tp_price, qty=qty_close,
                notional=qty_close * tp_price,
                pnl=partial_pnl, reason=f"TP{i+1}",
                dca_level=pos.get("dca_count", 0),
                entry_seed=seed, pnl_pct=partial_pct,
            )
            pos["tp_done"].append(i)
            pos["total_size"] -= qty_close
            if full_close:
                state["ct"] = None
                return True
    return False
