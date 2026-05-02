"""
OKX 거래소 래퍼 (CCXT)

기능:
  - 인증 및 연결 검증
  - 잔고/포지션 조회
  - 주문 생성/취소 (DRY_RUN 시 로그만)
  - 양방향(hedge) 모드 + 크로스 마진 가정
"""
import re
from typing import Optional

import ccxt

from live_bot import config, notifier


def _sanitize_clord_id(raw: Optional[str]) -> Optional[str]:
    """OKX clOrdId 규격(영숫자 1~32자)에 맞게 정리."""
    if not raw:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw)[:32]
    return cleaned or None


class OKXClient:
    def __init__(self) -> None:
        if not config.OKX_API_KEY:
            raise RuntimeError(".env.secret에 OKX_API_KEY가 없습니다.")

        self.ex = ccxt.okx({
            "apiKey":         config.OKX_API_KEY,
            "secret":         config.OKX_API_SECRET,
            "password":       config.OKX_PASSPHRASE,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",   # 영구 선물
            },
        })
        # 양방향 모드는 OKX에서 사용자가 거래소 UI에서 미리 설정해야 함
        # CCXT로는 set_position_mode() 가능하지만 잔여 주문 있으면 실패함

    # ── 인증/연결 ────────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            self.ex.fetch_time()
            return True
        except Exception as e:
            notifier.error(f"OKX 연결 실패: {e}")
            return False

    def fetch_balance_usdt(self) -> float:
        """USDT 잔고 (가용 + 포지션 마진)"""
        try:
            bal = self.ex.fetch_balance({"type": "swap"})
            return float(bal.get("USDT", {}).get("total", 0.0))
        except Exception as e:
            notifier.error(f"잔고 조회 실패: {e}")
            return 0.0

    # ── 시세 ────────────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list:
        """최근 N개 봉 (timestamp ms, O, H, L, C, V)"""
        return self.ex.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self.ex.fetch_ticker(symbol)
        return float(ticker["last"])

    # ── 포지션 ──────────────────────────────────────────────────────
    def fetch_positions(self, symbol: str) -> dict:
        """
        양방향 모드에서 long/short 포지션 분리 반환:
          {"long": {...} | None, "short": {...} | None}
        """
        try:
            poses = self.ex.fetch_positions([symbol])
        except Exception as e:
            notifier.error(f"포지션 조회 실패: {e}")
            return {"long": None, "short": None}

        result = {"long": None, "short": None}
        for p in poses:
            side = p.get("side") or p.get("info", {}).get("posSide")
            contracts = float(p.get("contracts", 0) or 0)
            if contracts == 0:
                continue
            if side == "long":
                result["long"] = p
            elif side == "short":
                result["short"] = p
        return result

    # ── 주문 ────────────────────────────────────────────────────────
    def create_market_order(
        self,
        symbol: str,
        side: str,            # "buy" or "sell"
        amount: float,        # 계약 수량 (base 단위)
        pos_side: str,        # "long" or "short" (양방향 모드)
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        시장가 주문. DRY_RUN 시 가짜 주문 dict 반환.

        OKX 양방향 모드 파라미터:
          posSide = "long" or "short"
          reduceOnly = 청산 주문일 때 True
        """
        params = {"posSide": pos_side, "tdMode": "cross"}
        if reduce_only:
            params["reduceOnly"] = True
        safe_id = _sanitize_clord_id(client_order_id)
        if safe_id:
            params["clOrdId"] = safe_id

        action = f"{side.upper()} {amount} {symbol} (posSide={pos_side}, reduceOnly={reduce_only})"

        if config.DRY_RUN:
            notifier.trade(f"[DRY-RUN] 주문 시뮬: {action}")
            return {
                "id": f"DRY-{client_order_id or 'auto'}",
                "status": "closed",
                "amount": amount,
                "side": side,
                "symbol": symbol,
                "info": {"dry_run": True, "params": params},
            }

        try:
            order = self.ex.create_order(symbol, "market", side, amount, params=params)
            notifier.trade(f"실주문 체결: {action} → id={order.get('id')}")
            return order
        except Exception as e:
            notifier.error(f"주문 실패: {action} | {e}")
            raise

    def close_all_positions(self, symbol: str) -> None:
        """긴급 청산: 양방향 long/short 모두 시장가 청산"""
        poses = self.fetch_positions(symbol)
        for side_name, pos in poses.items():
            if not pos:
                continue
            qty = float(pos["contracts"])
            close_side = "sell" if side_name == "long" else "buy"
            notifier.alert(f"긴급 청산: {symbol} {side_name} {qty}")
            self.create_market_order(
                symbol, close_side, qty, pos_side=side_name,
                reduce_only=True, client_order_id="emergency_close",
            )
