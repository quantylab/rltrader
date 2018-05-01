import time
import datetime
import locale
import logging
import os
import platform

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Strategy
# DEBUG = True
DEBUG = False
DEMO = True
# DEMO = False
TRANSACTION_TERM = 2  # 2 seconds
PROCESSING_TERM = 2  # 2 seconds
MARKET_WAIT_TERM = 10  # 10 seconds
MAX_TARGET_STOCK_PRICE = 500000
MAX_BUY_PRICE_AGG = 1000000
MAX_BUY_PRICE_DEF = 500000
BUY_UNIT_AGG = 500000
BUY_UNIT_DEF = 100000
TGT_TOP_DIFF = 10
TGT_BOTTOM_DIFF = -3
MIN_PRICE_VOLUME = 10000 * 10000
# Number of Holdings
MAX_NUM_HOLDINGS_AGG = 12
MAX_NUM_HOLDINGS_DEF = 5
# MAX_NUM_HOLDINGS_DEF = 0
# Monitoring Stocks
MAX_STOCKS_MONITOR_ITR = 5 # Each of KOSDAQ and KOSPI
FIVEMIN_INCDEC_RATE = 0.025


# Settings for Server/
SERVER_ADDR = "localhost"
SERVER_PORT = 8000
SERVER_URL = "http://%s:%s" % (SERVER_ADDR, SERVER_PORT)
SERVER_API_URL = "http://%s:%s/api" % (SERVER_ADDR, SERVER_PORT)
SERVER_WS_URL = "ws://%s:%s/ws" % (SERVER_ADDR, SERVER_PORT)


# Settings for Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Settings for Templates
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# Settings for Static
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_URL = "/static/"


# Settings for Data
DATA_DIR = os.path.join(BASE_DIR, "database")


# Date Time Format
timestr = None
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"


# 로케일 설정
if 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')


# Settings on Logging
def get_today_str():
    today = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str


def get_time_str():
    global timestr
    timestr = datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)
    return timestr
