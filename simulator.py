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