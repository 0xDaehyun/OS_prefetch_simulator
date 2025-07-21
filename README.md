# 📱 macOS App Prefetch Simulator

사용자의 앱 전환 패턴을 학습하여 다음에 실행할 가능성이 높은 앱을 예측하고 백그라운드로 미리 실행(prefetch)하는 macOS 전용 Python 시뮬레이터입니다.

---

## ✅ 기능

- 현재 포커스된 앱 감지 (AppleScript 사용)
- 앱 전이 패턴 학습 (Logistic Regression 기반)
- 다음 실행 앱 예측 및 백그라운드 실행 (`open -g`)
- 24시간치 랜덤 시퀀스를 기반으로 한 가상 로그 생성 (`simulate_logs`)

---

## 🧱 환경 요구사항

- macOS
- Python 3.8 이상

### 필수 패키지 설치

```bash
pip3 install numpy scikit-learn
