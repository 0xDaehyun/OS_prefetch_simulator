import subprocess
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

class AppPrefetcher:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_app = None
        self.model = None
        self.app_to_idx = {}
        self.idx_to_app = {}

    def update_model(self, prev_app, next_app):
        if prev_app and next_app:
            self.transition_counts[prev_app][next_app] += 1
            self.train_model()

    def train_model(self):
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
            classes = np.unique(y_arr)
            if len(classes) < 2:
                return
            clf = LogisticRegression()
            clf.fit(X_arr, y_arr)
            self.model = clf

    def predict_next(self, current_app):
        if not self.model or current_app not in self.app_to_idx:
            return None
        idx_cur = self.app_to_idx[current_app]
        pred_idx = self.model.predict([[idx_cur]])[0]
        return self.idx_to_app.get(pred_idx)

    def prefetch(self, app_name):
        try:
            subprocess.run(["open", "-a", app_name, "-g"], check=True)
            print(f"[Prefetch] {app_name}")
        except subprocess.CalledProcessError:
            print(f"[Error] '{app_name}' 실행 실패")

    def run(self, interval=2.0, get_active_app_func=None):
        """
        interval 초마다 포커스 앱 검사 → 모델 갱신 → 예측 → 프리패칭
        get_active_app_func: 외부에서 앱 이름을 얻어오는 함수를 주입
        """
        print("=== App Prefetcher 시작 ===")
        if get_active_app_func is None:
            from utils import get_active_app
            get_active_app_func = get_active_app

        while True:
            active_app = get_active_app_func()
            self.update_model(self.last_app, active_app)
            print(f"[Debug] Transition: {self.last_app} -> {active_app}")
            pred = self.predict_next(active_app)
            print(f"[Debug] Predicted next app: {pred}")
            if pred and pred != active_app:
                self.prefetch(pred)
            self.last_app = active_app
            import time; time.sleep(interval)