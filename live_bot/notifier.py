"""
로그 + 알림 (콘솔 + 파일, 텔레그램 옵션)

- INFO: 일반 동작 (콘솔/파일만, 디스코드 X)
- TRADE: 진입/청산/DCA (디스코드 — 표준 prefix)
- WARN: 안전 한도 근접 (콘솔/파일만)
- ERROR: 예외/실패 (콘솔/파일만)
- ALERT: 즉시 주의 (kill switch, 한도 도달) — 디스코드

디스코드 송신은 shared/discord_notify로 위임한다. INFO/WARN/ERROR/봇 시작/하트비트는
디스코드로 보내지 않는다.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from live_bot.config import LOG_DIR

# shared 어댑터 경로 확보
_p = Path(__file__).resolve()
for _parent in _p.parents:
    if (_parent / "shared" / "discord_notify.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
try:
    from shared.discord_notify import notify_alert as _dn_alert, notify_info as _dn_info
    _SHARED_OK = True
except ImportError:
    _SHARED_OK = False


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("multist_live")
    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔
    h_con = logging.StreamHandler(sys.stdout)
    h_con.setFormatter(fmt)
    log.addHandler(h_con)

    # 파일 (일자별)
    log_file = LOG_DIR / f"bot_{datetime.utcnow():%Y%m%d}.log"
    h_file = logging.FileHandler(log_file, encoding="utf-8")
    h_file.setFormatter(fmt)
    log.addHandler(h_file)

    return log


_log = _setup_logger()


def info(msg: str) -> None:
    _log.info(msg)


def warn(msg: str) -> None:
    _log.warning(f"[WARN] {msg}")


def error(msg: str) -> None:
    _log.error(f"[ERROR] {msg}")


def alert(msg: str) -> None:
    """긴급 알림 — 표준 디스코드 어댑터로 전송 + 텔레그램 시도"""
    _log.error(f"[ALERT] {msg}")
    _try_telegram(f"🚨 {msg}")
    if _SHARED_OK:
        _dn_alert("MultiST", msg)


def trade(msg: str) -> None:
    """거래 발생 알림 — 표준 디스코드 어댑터(info)로 전송.

    호출자가 만든 텍스트(예: '메인 LONG 진입 시도: qty=...')를 그대로 본문으로 보낸다.
    진입/청산 표준 포맷이 필요한 경우 호출자가 shared.discord_notify.notify_open/close를
    직접 호출하면 된다.
    """
    _log.info(f"[TRADE] {msg}")
    if _SHARED_OK:
        _dn_info("MultiST", msg)


def _try_telegram(msg: str) -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg}).encode()
        urllib.request.urlopen(url, data, timeout=5)
    except Exception as e:
        _log.error(f"텔레그램 전송 실패: {e}")


def _try_discord(msg: str) -> None:
    """[deprecated] shared.discord_notify로 위임됨. 직접 호출되더라도 표준 어댑터 경유."""
    if _SHARED_OK:
        _dn_info("MultiST", msg)
