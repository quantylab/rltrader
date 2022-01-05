import sys
import os
import locale
import platform


# 경로 설정
BASE_DIR = os.path.dirname(sys.modules['__main__'].__file__)


# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
