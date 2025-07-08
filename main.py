import time
import subprocess
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression ## 해당 스텍을 사용함으로써
##sklearn.linear_model 에서 제공하는  머신러닝 분류 알고리즘 중 하나인 , Logistic Regression 클래스를 이용하여 ,
def get_active_app():
    """
    현재 포커스된 앱 이름을 AppleScript로 가져옵니다.
    """
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    try:
        out = subprocess.check_output(["osascript", "-e", script])
        return out.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None

# 가상 로그 시뮬레이션 함수
def simulate_logs(pfetcher, duration_hours=24, interval_min=5):
    """
    duration_hours 동안 interval_min 분 단위로
    랜덤으로 선택된 앱 시퀀스를 모델에 feed 합니다.
    """
    import random
    apps = list(pfetcher.app_to_idx.keys()) or ["Safari", "PyCharm", "Terminal", "ChatGPT", "KakaoTalk"]
    prev = None
    for _ in range(int(duration_hours * 60 / interval_min)):
        curr = random.choice(apps)
        pfetcher.update_model(prev, curr)
        prev = curr

class AppPrefetcher:
    def __init__(self):
        # 이전 앱 → 다음 앱 전이 횟수 기록
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_app = None
        # ML 모델 초기화
        self.model = None
        self.app_to_idx = {}
        self.idx_to_app = {}

    def update_model(self, prev_app, next_app):
        """이전 앱(prev_app)에서 next_app으로의 전이 횟수 증가"""
        if prev_app and next_app:
            self.transition_counts[prev_app][next_app] += 1
            # ML 모델 재학습
            self.train_model()

    def train_model(self):
        """
        transition_counts를 기반으로 로지스틱 회귀 모델을 학습합니다.
        """
        # 앱 목록 구성
        apps = set(self.transition_counts.keys())
        for prev in self.transition_counts:
            apps.update(self.transition_counts[prev].keys())
        apps = list(apps)
        # 인덱스 매핑
        self.app_to_idx = {app: i for i, app in enumerate(apps)}
        self.idx_to_app = {i: app for app, i in self.app_to_idx.items()}
        # 학습 데이터 준비
        X, y = [], []
        for prev_app, next_dict in self.transition_counts.items():
            for next_app, count in next_dict.items():
                idx_prev = self.app_to_idx[prev_app]
                idx_next = self.app_to_idx[next_app]
                X.extend([[idx_prev]] * count)
                y.extend([idx_next] * count)
        # 모델 학습
        if X and len(apps) >= 2:
            X_arr = np.array(X)
            y_arr = np.array(y)
            # 최소 두 개 클래스가 있어야 모델 학습
            classes = np.unique(y_arr)
            if len(classes) < 2:
                return
            clf = LogisticRegression()
            clf.fit(X_arr, y_arr)
            self.model = clf

    def predict_next(self, current_app):
        """
        ML 모델을 이용해 다음 앱을 예측합니다.
        """
        if not self.model or current_app not in self.app_to_idx:
            return None
        idx_cur = self.app_to_idx[current_app]
        pred_idx = self.model.predict([[idx_cur]])[0]
        return self.idx_to_app.get(pred_idx)

    def prefetch(self, app_name):
        """
        macOS 'open' 명령으로 앱을 백그라운드(prefetch) 실행
        -g 옵션: 포어그라운드로 전환하지 않음
        """
        try:
            subprocess.run(["open", "-a", app_name, "-g"], check=True)
            print(f"[Prefetch] {app_name}")
        except subprocess.CalledProcessError:
            print(f"[Error] '{app_name}' 실행 실패")

    def run(self, interval=2.0):
        """메인 루프: interval 초마다 포커스 앱 검사 → 모델 갱신 → 예측 → 프리패칭"""
        print("=== App Prefetcher 시작 ===")
        while True:
            # 현재 포커스된 앱 가져오기 (AppleScript 이용)
            active_app = get_active_app()
            # 모델 갱신
            self.update_model(self.last_app, active_app)
            print(f"[Debug] Transition: {self.last_app} -> {active_app}")
            # 예측 & 프리패칭
            pred = self.predict_next(active_app)
            print(f"[Debug] Predicted next app: {pred}")
            if pred and pred != active_app:
                self.prefetch(pred)
            # 다음 사이클을 위해 저장
            self.last_app = active_app
            time.sleep(interval)

if __name__ == "__main__":
    pfetcher = AppPrefetcher()
    # 24시간 가상 로그 시뮬레이션 (5분 간격)
    simulate_logs(pfetcher, duration_hours=24, interval_min=5)
    try:
        pfetcher.run(interval=1.5)
    except KeyboardInterrupt:
        print("\n프로그램 종료")


# cd ~/PycharmProjects/os_prefetch_simulator
        # pip3 install numpy scikit-learn
        # 실행: python3 main.py