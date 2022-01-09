@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 카카오: 035720
@REM LG화학: 051910
@REM 현대차: 005380
@REM 셀트리온: 068270
@REM POSCO: 005490

@REM python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode train --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode train --ver v3 --name 005490 --stock_code 005490 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 005490 --stock_code 005490 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name 005490 --stock_code 005490 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name 005490 --stock_code 005490 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode train --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231
