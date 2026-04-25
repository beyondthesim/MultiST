"""
로그 + 알림 (콘솔 + 파일, 텔레그램 옵션)

- INFO: 일반 동작
- TRADE: 진입/청산/DCA
- WARN: 안전 한도 근접
- ERROR: 예외/실패
- ALERT: 즉시 주의 (kill switch, 한도 도달)
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from live_bot.config import LOG_DIR


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
    """긴급 알림 — 텔레그램/디스코드 모두 시도"""
    _log.error(f"[ALERT] {msg}")
    _try_telegram(f"🚨 {msg}")
    _try_discord(f":rotating_light: **ALERT** {msg}")


def trade(msg: str) -> None:
    """거래 발생 알림 — 디스코드로도 전송"""
    _log.info(f"[TRADE] {msg}")
    _try_discord(f":moneybag: {msg}")


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
    """DISCORD_URL 환경변수에 웹훅 URL이 있으면 메시지 전송"""
    url = os.getenv("DISCORD_URL", "").strip().strip('"')
    if not url:
        return
    try:
        import json as _json
        import urllib.request
        data = _json.dumps({"content": msg[:1900]}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent":   "MultiST-LiveBot/1.0 (Python urllib)",
            },
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        _log.error(f"디스코드 전송 실패: {e}")
