# MultiST 라이브 봇 운영 가이드

## 사전 점검 (필수)

### 1. OKX 거래소 셋업
- [ ] **포지션 모드**: Hedge Mode (양방향) — Account Settings → Position Mode
- [ ] **마진 모드**: Cross Margin (PI/USDT:USDT 페어에서 설정)
- [ ] **레버리지**: PI/USDT:USDT 20x
- [ ] **API Key 발급**: Trade + Read 권한, **출금 OFF**, IP 화이트리스트
- [ ] **잔고 입금**: 시작 자본 (권장 첫 1주 $100~500)

### 2. 환경 파일
```bash
cp live_bot/.env.example .env.secret
# .env.secret 열어서 OKX_API_KEY, SECRET, PASSPHRASE 입력
```

`.gitignore`에 `.env.secret` 추가되어 있는지 확인하세요.

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

---

## 실행 절차 (단계별)

### Step 1 — DRY-RUN 모드 (1~2일)

주문 안 내고 신호만 출력. 백테스트 신호와 일치 확인.

```bash
python -m live_bot.main
```

콘솔/로그 확인:
- `[DRY-RUN] 주문 시뮬: ...` 로그가 정상 출력되는지
- 새 봉 도착 시 신호가 백테스트 결과와 일치하는지
- 12분마다 봉이 잘 도착하는지

로그 위치: `live_bot/logs/bot_YYYYMMDD.log`

### Step 2 — 소액 실거래 (1주)

`.env.secret` 수정:
```
LIVE_MODE=true
MAX_DAILY_LOSS_PCT=10
MAX_DD_PCT=20
```

소액($100~500) 잔고 상태에서:
```bash
python -m live_bot.main
```

체크 사항:
- 첫 실주문이 거래소 UI에서 정확히 체결되는지
- 봇 상태 ↔ 거래소 포지션 일치 (포지션 동기화 로그)
- SL/TP 정상 작동
- DCA 트리거 정상 작동

### Step 3 — 본격 운영

검증 OK 후 자본 증액. MDD 한도는 보수적으로 유지(20~30%).

---

## 안전 장치

| 장치 | 동작 |
|------|------|
| **DRY_RUN** | `LIVE_MODE=false`면 주문 시뮬만, 실거래 X |
| **Kill switch** | `live_bot/kill.txt` 파일 생성 → 다음 루프에서 모든 포지션 청산 후 정지 |
| **일일 손실 한도** | -10% 도달 시 당일 신규 진입 차단 (보유 포지션 유지) |
| **MDD 한도** | -30% 도달 시 모든 포지션 청산 + 봇 영구 정지 |
| **Halt 상태** | 정지 후 재시작하려면 `live_bot/state/bot_state.json`에서 `halted: false`로 수정 |

### 비상 정지
```bash
echo "stop" > live_bot/kill.txt
```
다음 루프(최대 30초)에 모든 포지션 시장가 청산 후 봇 정지.

### 봇 종료 (포지션 유지)
- `Ctrl+C`: 봇만 종료, 거래소 포지션은 그대로 유지

---

## 상태 파일

`live_bot/state/bot_state.json`:
- `initial_capital`: 첫 실행 시 자본
- `peak_equity`: MDD 추적용 최고점
- `daily_start_equity`: 당일 시작 자본
- `last_bar_ts`: 마지막 처리한 봉 시각 (중복 신호 방지)
- `main`, `ct`: 진행 중 포지션 정보
- `halted`: 봇 정지 여부

**상태 리셋이 필요한 경우** (자본 변경, 봇 재시작 등):
```bash
rm live_bot/state/bot_state.json
```

---

## 트러블슈팅

### "OKX 연결 실패"
- API Key 권한 확인 (Trade + Read)
- IP 화이트리스트에 현재 IP 추가됐는지
- API Key가 mainnet용인지 (demo 아님)

### "포지션 불일치 LONG/SHORT"
- 거래소 UI에서 수동 거래했거나 봇 외부 주문이 있었음
- 봇 정지 후 `bot_state.json` 수정해서 동기화

### "주문 실패"
- 마진 부족 (자본보다 큰 명목 진입 시도) → `main_position_pct`/`ct_position_pct` 조정
- 레버리지 한도 초과 → 거래소에서 PI/USDT 레버리지 상향
- 최소 주문 수량 미달 → 자본 증액

---

## 알려진 한계

1. **봉 close 시점에만 SL/TP 판정** — 봉 사이 wick으로 SL이 의도보다 늦게 발동될 수 있음. 향후 OKX algorithmic order로 개선 가능.
2. **양방향 모드 1슬롯 제한** — 메인 LONG + CT LONG 동시는 거래소에서 합쳐짐. 봇은 logical 분리 추적하지만 거래소는 단일 포지션으로 본다. 메인+CT가 같은 방향일 때 주의.
3. **백테스트 결과 ≠ 실거래** — 슬리피지·latency·partial fill로 30~50% 수익 감소 가능.
