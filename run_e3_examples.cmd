@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net dnn --start_date 20180101 --end_date 20191231
