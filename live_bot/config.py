"""
라이브 봇 설정

핵심 안전 토글:
  DRY_RUN: True = 주문 실행 안 함 (로그만), False = 실제 주문
  실거래 시작은 명시적으로 환경변수 LIVE_MODE=true 설정 시에만.

환경변수 (.env.secret):
  OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE
  LIVE_MODE (기본 false)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (선택)
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# .env.secret 우선 로드
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.secret")
load_dotenv(ROOT / ".env")

# ── 기본 안전 모드 ─────────────────────────────────────────────────────
# LIVE_MODE 환경변수가 명시적으로 "true"가 아니면 DRY_RUN 강제
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"
DRY_RUN   = not LIVE_MODE

# ── 거래소 인증 ────────────────────────────────────────────────────────
OKX_API_KEY    = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ── 안전 한도 ─────────────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "10"))   # 하루 -10% 도달 시 정지
MAX_DD_PCT         = float(os.getenv("MAX_DD_PCT", "30"))            # 누적 MDD 30% 도달 시 정지
KILL_SWITCH_FILE   = ROOT / "live_bot" / "kill.txt"                  # 이 파일 생성 시 즉시 청산+정지

# ── 봇 동작 ────────────────────────────────────────────────────────────
LOOP_INTERVAL_SEC  = int(os.getenv("LOOP_INTERVAL_SEC", "30"))       # 메인 루프 주기
SYNC_POSITIONS_SEC = int(os.getenv("SYNC_POSITIONS_SEC", "300"))     # 5분마다 거래소 포지션 동기화
WARMUP_BARS        = int(os.getenv("WARMUP_BARS", "300"))            # 신호 계산용 워밍업 봉수

# ── 상태 파일 경로 ─────────────────────────────────────────────────────
STATE_DIR = ROOT / "live_bot" / "state"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "bot_state.json"
LOG_DIR    = ROOT / "live_bot" / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── 백테스트 파라미터 재사용 ───────────────────────────────────────────
PARAMS_FILE = ROOT / "config" / "params.json"


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return json.load(f)


def summary() -> str:
    return (
        f"\n{'='*60}\n"
        f"  MultiST Live Bot 설정\n"
        f"{'='*60}\n"
        f"  모드:           {'LIVE [실거래]' if LIVE_MODE else 'DRY-RUN [시뮬]'}\n"
        f"  API key:        {'설정됨' if OKX_API_KEY else '[없음] .env.secret 확인 필요'}\n"
        f"  일일 손실 한도: -{MAX_DAILY_LOSS_PCT}%\n"
        f"  MDD 한도:       {MAX_DD_PCT}%\n"
        f"  Kill switch:    {KILL_SWITCH_FILE}\n"
        f"  루프 주기:      {LOOP_INTERVAL_SEC}s\n"
        f"  포지션 동기화:  {SYNC_POSITIONS_SEC}s\n"
        f"{'='*60}"
    )
