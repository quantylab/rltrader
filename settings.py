import locale
import os
import platform


# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 로케일 설정
try:
    if 'Linux' in platform.system() or 'Darwin' in platform.system():
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    elif 'Windows' in platform.system():
        locale.setlocale(locale.LC_ALL, '')
except Exception as e:
    print('[settings] e:', e)
