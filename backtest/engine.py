"""
백테스트 엔진

TradingView process_orders_on_close=true 모방:
  - 신호 발생 봉의 close 가격에 체결
  - TP는 다음 봉부터 high/low로 체크 → limit 가격에 체결
  - SL은 close 기준 소프트 SL (봉 종가가 SL 이하일 때 종가에 청산)

포지션 크기:
  - 메인 전략: 전체 자산의 (100 - ct_equity_pct)%
  - 역추세 전략: 전체 자산의 ct_equity_pct% (독립 운용, DCA 지원)
수수료: commission_pct % (진입/청산 각각)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class PartialClose:
    reason: str
    price: float
    units: float
    gross_value: float
    commission: float

    @property
    def net_value(self) -> float:
        return self.gross_value - self.commission


@dataclass
class Trade:
    direction: int          # 1=Long, -1=Short
    entry_price: float      # 평균단가 (DCA 후 갱신)
    entry_time: pd.Timestamp
    entry_equity: float     # 총 투입 자본 (DCA 후 누적)
    commission_rate: float
    strategy: str = "main"  # "main" | "ct"

    tp_levels: list = field(default_factory=list)

    units: float = field(init=False)
    remaining_units: float = field(init=False)
    entry_commission: float = field(init=False)
    last_entry_price: float = field(init=False)   # 마지막 진입가 (DCA 가격 트리거용)
    closes: list = field(default_factory=list)
    exit_time: Optional[pd.Timestamp] = None
    close_reason: str = ""
    dca_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.units = self.entry_equity / self.entry_price
        self.remaining_units = self.units
        self.entry_commission = self.entry_equity * self.commission_rate
        self.last_entry_price = self.entry_price

    # ── DCA 추가 진입 ─────────────────────────────────────────────────
    def add_entry(
        self,
        add_price: float,
        add_equity: float,
        tp_pcts: list[dict],
    ) -> None:
        """DCA 추가 진입: 평균단가 재계산 후 남은 TP 재설정"""
        add_units = add_equity / add_price
        add_commission = add_equity * self.commission_rate

        total_cost = self.remaining_units * self.entry_price + add_units * add_price
        self.remaining_units += add_units
        self.units += add_units
        self.entry_price = total_cost / self.remaining_units
        self.entry_equity += add_equity
        self.entry_commission += add_commission
        self.last_entry_price = add_price   # 마지막 DCA 가격 기록
        self.dca_count += 1

        # 새 평균단가 기준으로 TP 재계산
        self.tp_levels = [
            {
                "price":   self.entry_price * (1 + self.direction * tp["pct"]),
                "qty_pct": tp["qty_pct"],
            }
            for tp in tp_pcts
        ]
        self.tp_levels.sort(key=lambda x: x["price"], reverse=(self.direction == -1))

    # ── 청산 ─────────────────────────────────────────────────────────
    def close_partial(
        self,
        close_price: float,
        qty_pct_of_remaining: float,
        reason: str,
        ts: Optional[pd.Timestamp] = None,
    ) -> float:
        units_to_close = self.remaining_units * (qty_pct_of_remaining / 100.0)
        if units_to_close <= 0:
            return 0.0

        gross_value = units_to_close * close_price
        commission  = gross_value * self.commission_rate

        pc = PartialClose(
            reason      = reason,
            price       = close_price,
            units       = units_to_close,
            gross_value = gross_value,
            commission  = commission,
        )
        self.closes.append(pc)
        self.remaining_units -= units_to_close
        if ts:
            self.exit_time = ts
        return gross_value - commission

    def close_full(
        self,
        close_price: float,
        reason: str,
        ts: Optional[pd.Timestamp] = None,
    ) -> float:
        self.close_reason = reason
        return self.close_partial(close_price, 100.0, reason, ts)

    # ── 계산 ─────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self.remaining_units > 1e-12

    @property
    def units_closed(self) -> float:
        return sum(c.units for c in self.closes)

    @property
    def total_exit_proceeds(self) -> float:
        return sum(c.gross_value for c in self.closes)

    @property
    def total_exit_commissions(self) -> float:
        return sum(c.commission for c in self.closes)

    @property
    def cost_basis(self) -> float:
        return self.units_closed * self.entry_price

    @property
    def gross_pnl(self) -> float:
        if self.direction == 1:
            return self.total_exit_proceeds - self.cost_basis
        else:
            return self.cost_basis - self.total_exit_proceeds

    @property
    def total_commission(self) -> float:
        return self.entry_commission + self.total_exit_commissions

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_commission

    @property
    def net_pnl_pct(self) -> float:
        return self.net_pnl / self.entry_equity * 100.0

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def avg_exit_price(self) -> float:
        if not self.closes:
            return self.entry_price
        total_value = sum(c.gross_value for c in self.closes)
        total_units = sum(c.units for c in self.closes)
        return total_value / total_units if total_units > 0 else self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open:
            return 0.0
        cost = self.remaining_units * self.entry_price
        value = self.remaining_units * current_price
        if self.direction == 1:
            return value - cost
        else:
            return cost - value

    def to_dict(self) -> dict:
        return {
            "strategy":       self.strategy,
            "direction":      "LONG" if self.direction == 1 else "SHORT",
            "entry_time":     self.entry_time,
            "exit_time":      self.exit_time,
            "entry_price":    self.entry_price,
            "avg_exit_price": self.avg_exit_price,
            "entry_equity":   self.entry_equity,
            "net_pnl":        self.net_pnl,
            "net_pnl_pct":   self.net_pnl_pct,
            "is_winner":      self.is_winner,
            "close_reason":   self.close_reason,
            "tp_count":       sum(1 for c in self.closes if c.reason.startswith("TP")),
            "n_closes":       len(self.closes),
            "dca_count":      self.dca_count,
        }


class BacktestEngine:
    def __init__(self, params: dict):
        self.params = params
        self.commission_rate = params["commission_pct"] / 100.0

    # ── 메인 포지션 오픈 ──────────────────────────────────────────────
    def _open_trade(
        self,
        direction: int,
        bar: dict,
        ts: pd.Timestamp,
        equity: float,
    ) -> Trade:
        tp_key = "long_tp" if direction == 1 else "short_tp"
        tp_list = [
            {
                "price":   bar["close"] * (1 + direction * tp["pct"]),
                "qty_pct": tp["qty_pct"],
            }
            for tp in self.params[tp_key]
        ]
        tp_list.sort(key=lambda x: x["price"], reverse=(direction == -1))

        return Trade(
            direction       = direction,
            entry_price     = bar["close"],
            entry_time      = ts,
            entry_equity    = equity,
            commission_rate = self.commission_rate,
            tp_levels       = tp_list,
            strategy        = "main",
        )

    # ── 역추세 포지션 오픈 ────────────────────────────────────────────
    def _open_ct_trade(
        self,
        direction: int,
        bar: dict,
        ts: pd.Timestamp,
        ct_equity: float,
    ) -> tuple["Trade", float]:
        """
        CT 포지션 오픈. (Trade, per_unit_size) 반환.

        Safe9 DCA 가중치 방식:
          dca_weights = [1,1,1,1,2,3,5,8,13] (파인스크립트 Safe9)
          total_weight = 1(첫진입) + sum(weights[:max_dca])
          per_unit = ct_equity / total_weight
          첫 진입: per_unit × 1
          k차 DCA: per_unit × dca_weights[k]
        """
        ct_cfg      = self.params["counter_trend"]
        max_dca     = ct_cfg.get("max_dca", 4)
        dca_weights = ct_cfg.get("dca_weights", [1, 1, 1, 1, 2, 3, 5, 8, 13])
        used_w      = dca_weights[:max_dca]
        total_w     = 1 + sum(used_w)
        per_unit    = ct_equity / total_w  # 1 가중치 단위 크기

        # ct_long_tp / ct_short_tp 기준으로 TP 설정 (없으면 all_close_pct 단일 TP)
        tp_key    = "ct_long_tp" if direction == 1 else "ct_short_tp"
        fallback  = [{"pct": ct_cfg.get("all_close_pct", 0.04), "qty_pct": 100}]
        tp_config = ct_cfg.get(tp_key, fallback)
        ep        = bar["close"]
        tp_list   = [
            {"price": ep * (1 + direction * tp["pct"]), "qty_pct": tp["qty_pct"]}
            for tp in tp_config
        ]

        trade = Trade(
            direction       = direction,
            entry_price     = bar["close"],
            entry_time      = ts,
            entry_equity    = per_unit,
            commission_rate = self.commission_rate,
            tp_levels       = tp_list,
            strategy        = "ct",
        )
        return trade, per_unit

    # ── 포지션 TP 처리 (공통) ─────────────────────────────────────────
    @staticmethod
    def _process_tp(position: Trade, bar: dict, ts: pd.Timestamp) -> None:
        remaining_tps = []
        for tp in position.tp_levels:
            if position.direction == 1:
                hit = bar["high"] >= tp["price"]
            else:
                hit = bar["low"] <= tp["price"]

            if hit and position.is_open:
                position.close_partial(tp["price"], tp["qty_pct"], "TP", ts)
            else:
                remaining_tps.append(tp)
        position.tp_levels = remaining_tps
        if not position.is_open and not position.close_reason:
            position.close_reason = "TP"

    # ── 메인 백테스트 루프 ───────────────────────────────────────────
    def run(
        self, df: pd.DataFrame
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        백테스트 실행

        Returns
        -------
        main_trades : 메인 전략 거래 dict 목록
        ct_trades   : 역추세 전략 거래 dict 목록
        equity_curve: [{"timestamp": ..., "equity": ...}, ...] (합산)
        """
        ct_cfg     = self.params.get("counter_trend", {})
        ct_enabled = ct_cfg.get("enabled", False)

        ct_equity_pct = ct_cfg.get("equity_pct", 30) / 100.0 if ct_enabled else 0.0
        initial       = self.params["initial_capital"]

        # equity_pct=0 → 레버리지 모드: 자본 분할 없이 메인·CT 모두 초기 자본 전체로 포지션
        # ct_size_pct: 레버리지 모드에서 CT 진입 사이즈를 메인 대비 축소 (기본 100% = 메인과 동일)
        ct_size_pct = ct_cfg.get("ct_size_pct", 100) / 100.0 if ct_enabled else 1.0
        if ct_enabled and ct_equity_pct == 0:
            main_equity    = initial
            ct_equity      = 0.0
            ct_base_equity = initial * ct_size_pct
        else:
            main_equity    = initial * (1.0 - ct_equity_pct)
            ct_equity      = initial * ct_equity_pct
            ct_base_equity = initial * ct_equity_pct  # 포지션 사이징 기준 (고정)

        long_sl_pct  = self.params["long_sl_pct"]
        short_sl_pct = self.params["short_sl_pct"]
        ct_sl_long   = ct_cfg.get("sl_long_pct",  0.03)
        ct_sl_short  = ct_cfg.get("sl_short_pct", 0.03)

        ct_max_dca      = ct_cfg.get("max_dca", 4)
        ct_dca_weights  = ct_cfg.get("dca_weights", [1, 1, 1, 1, 2, 3, 5, 8, 13])
        ct_dca_price_pct      = ct_cfg.get("dca_price_pct", 0.013)
        ct_dca_require_div    = ct_cfg.get("dca_require_divergence", True)
        ct_all_close_pct      = ct_cfg.get("all_close_pct",    0.04)
        ct_safe_close_count   = ct_cfg.get("safe_close_count", 3)
        ct_safe_close_pct     = ct_cfg.get("safe_close_pct",   0.02)

        main_trades: list[dict] = []
        ct_trades:   list[dict] = []
        eq_curve:    list[dict] = []

        position:    Optional[Trade] = None
        ct_position: Optional[Trade] = None
        ct_per_entry: float = 0.0    # CT 진입당 고정 크기 (포지션 오픈 시 확정)

        bars       = df.to_dict("index")
        timestamps = list(df.index)
        weekday_only = self.params.get("weekday_only", False)

        for ts in timestamps:
            bar = bars[ts]
            is_weekday = ts.weekday() < 5

            # ── 메인 포지션 처리 ──────────────────────────────────────
            if position is not None:
                self._process_tp(position, bar, ts)

                if not position.is_open:
                    position.close_reason = "TP_FULL"
                    main_equity += position.net_pnl
                    main_trades.append(position.to_dict())
                    position = None

                if position is not None:
                    if position.direction == 1:
                        sl_hit = bar["close"] < position.entry_price * (1 - long_sl_pct)
                    else:
                        sl_hit = bar["close"] > position.entry_price * (1 + short_sl_pct)

                    if sl_hit:
                        position.close_full(bar["close"], "SL", ts)
                        main_equity += position.net_pnl
                        main_trades.append(position.to_dict())
                        position = None

                if position is not None:
                    st_flip = (
                        (position.direction == 1  and bar["close_long"])  or
                        (position.direction == -1 and bar["close_short"])
                    )
                    if st_flip:
                        position.close_full(bar["close"], "ST_FLIP", ts)
                        main_equity += position.net_pnl
                        main_trades.append(position.to_dict())
                        position = None

            # ── 역추세 포지션 처리 ────────────────────────────────────
            if ct_enabled and ct_position is not None:
                d = ct_position.direction
                avg_p = ct_position.entry_price

                # 1) 다단계 TP (ct_long_tp / ct_short_tp, _process_tp 공통 처리)
                if ct_position.tp_levels:
                    self._process_tp(ct_position, bar, ts)
                    if not ct_position.is_open:
                        ct_equity   += ct_position.net_pnl
                        ct_trades.append(ct_position.to_dict())
                        ct_position  = None
                        ct_per_entry = 0.0

                # 2) safe_close TP (DCA ≥ N회 후 낮은 TP로 빠른 탈출)
                if ct_position is not None and ct_position.dca_count >= ct_safe_close_count:
                    avg_p = ct_position.entry_price
                    if d == 1:
                        safe_price = avg_p * (1 + ct_safe_close_pct)
                        safe_hit   = bar["high"] >= safe_price
                    else:
                        safe_price = avg_p * (1 - ct_safe_close_pct)
                        safe_hit   = bar["low"] <= safe_price

                    if safe_hit:
                        ct_position.close_full(safe_price, "TP_SAFE", ts)
                        ct_equity   += ct_position.net_pnl
                        ct_trades.append(ct_position.to_dict())
                        ct_position  = None
                        ct_per_entry = 0.0

                # 3) 하드 SL (봉 내 low/high 터치 시 SL 가격에 청산)
                if ct_position is not None:
                    ct_sl_pct = ct_sl_long if d == 1 else ct_sl_short
                    if d == 1:
                        sl_price = ct_position.entry_price * (1 - ct_sl_pct)
                        sl_hit   = bar["low"] <= sl_price
                    else:
                        sl_price = ct_position.entry_price * (1 + ct_sl_pct)
                        sl_hit   = bar["high"] >= sl_price

                    if sl_hit:
                        ct_position.close_full(sl_price, "SL", ts)
                        ct_equity   += ct_position.net_pnl
                        ct_trades.append(ct_position.to_dict())
                        ct_position  = None
                        ct_per_entry = 0.0

                # 4) CT_EXIT (ST 역전 or RSI 회복)
                if ct_position is not None:
                    close_col = "ct_close_long" if d == 1 else "ct_close_short"
                    if bar.get(close_col, False):
                        ct_position.close_full(bar["close"], "CT_EXIT", ts)
                        ct_equity   += ct_position.net_pnl
                        ct_trades.append(ct_position.to_dict())
                        ct_position  = None
                        ct_per_entry = 0.0

            # ── 역추세 DCA 체크 (가격 기반 + RSI 다이버전스) ──────────
            if (ct_enabled and ct_position is not None
                    and ct_position.dca_count < ct_max_dca
                    and ct_per_entry > 0):
                next_idx = ct_position.dca_count
                d        = ct_position.direction

                # 가격 조건: 직전 매수가 대비 dca_price_pct% 추가 하락
                last_p = ct_position.last_entry_price
                if d == 1:
                    price_ok = bar["close"] < last_p * (1 - ct_dca_price_pct)
                else:
                    price_ok = bar["close"] > last_p * (1 + ct_dca_price_pct)

                # 다이버전스 조건 (선택)
                if ct_dca_require_div:
                    div_col = "rsi_bull_div" if d == 1 else "rsi_bear_div"
                    div_ok  = bool(bar.get(div_col, False))
                else:
                    div_ok = True

                if price_ok and div_ok:
                    # Safe9 가중치로 DCA 금액 결정
                    weight = ct_dca_weights[next_idx] if next_idx < len(ct_dca_weights) else 1
                    dca_amount = ct_per_entry * weight

                    # TP 리스트: 첫 진입과 동일한 방향별 TP 유지 (DCA 후에도 일관된 익절 전략)
                    tp_key   = "ct_long_tp" if d == 1 else "ct_short_tp"
                    fallback = [{"pct": ct_all_close_pct, "qty_pct": 100}]
                    tp_pcts  = ct_cfg.get(tp_key, fallback)
                    ct_position.add_entry(bar["close"], dca_amount, tp_pcts)

            # ── 동적 명목 사이징 (옵션: 자본의 % × 레버리지) ──────────
            # main_position_pct: 자본 대비 메인 명목 % (예: 200 = 자본 × 2)
            # ct_position_pct:   자본 대비 CT 풀 사이클 명목 % (DCA 10회 모두 시)
            main_pos_pct = self.params.get("main_position_pct", 0)
            ct_pos_pct   = ct_cfg.get("ct_position_pct", 0)
            current_eq   = main_equity + ct_equity
            main_open_size = current_eq * main_pos_pct / 100.0 if main_pos_pct > 0 else main_equity
            ct_open_base   = current_eq * ct_pos_pct   / 100.0 if ct_pos_pct   > 0 else ct_base_equity

            # ── 메인 신규 진입 ────────────────────────────────────────
            if position is None:
                can_enter = (not weekday_only) or is_weekday
                if can_enter:
                    if bar["long_entry"]:
                        position = self._open_trade(1, bar, ts, main_open_size)
                    elif bar["short_entry"]:
                        position = self._open_trade(-1, bar, ts, main_open_size)

            # ── 역추세 신규 진입 ──────────────────────────────────────
            if ct_enabled and ct_position is None:
                ct_can = (not weekday_only) or is_weekday
                if ct_can:
                    if bar.get("ct_long_entry", False):
                        ct_position, ct_per_entry = self._open_ct_trade(1, bar, ts, ct_open_base)
                    elif bar.get("ct_short_entry", False):
                        ct_position, ct_per_entry = self._open_ct_trade(-1, bar, ts, ct_open_base)

            # ── 자산 곡선 (메인 + CT 합산) ────────────────────────────
            total_eq = main_equity + ct_equity
            if position is not None:
                total_eq += position.unrealized_pnl(bar["close"])
            if ct_position is not None:
                total_eq += ct_position.unrealized_pnl(bar["close"])
            eq_curve.append({"timestamp": ts, "equity": total_eq})

        # 미결 포지션: 마지막 close로 강제 청산
        last_bar = bars[timestamps[-1]]
        if position is not None:
            position.close_full(last_bar["close"], "END", timestamps[-1])
            main_equity += position.net_pnl
            main_trades.append(position.to_dict())
        if ct_position is not None:
            ct_position.close_full(last_bar["close"], "END", timestamps[-1])
            ct_equity += ct_position.net_pnl
            ct_trades.append(ct_position.to_dict())

        return main_trades, ct_trades, eq_curve
