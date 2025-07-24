import subprocess

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