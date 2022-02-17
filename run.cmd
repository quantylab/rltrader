@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 현대차: 005380
@REM 카카오: 035720

python main.py --mode train --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode train --ver v3 --name 005380 --stock_code 005380 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode train --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20180101 --end_date 20191231

python main.py --mode test --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode test --ver v3 --name 005380 --stock_code 005380 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode test --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode update --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name 005380 --stock_code 005380 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode predict --ver v3 --name 005930 --stock_code 005930 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name 005380 --stock_code 005380 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name 035720 --stock_code 035720 --rl_method a2c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode train --ver v3 --name all --stock_code 005930 005380 035720 --rl_method a3c --net lstm --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name all --stock_code 005930 005380 035720 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode update --ver v3 --name all --stock_code 005930 005380 035720 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231
python main.py --mode predict --ver v3 --name all --stock_code 005930 005380 035720 --rl_method a3c --net lstm --start_date 20200101 --end_date 20211231

python main.py --mode test --ver v3 --name 005930a --stock_code 005930 --rl_method monkey --net monkey --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 005930b --stock_code 005930 --rl_method monkey --net monkey --start_date 20200101 --end_date 20211231
python main.py --mode test --ver v3 --name 005380a --stock_code 005380 --rl_method monkey --net monkey --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 005380b --stock_code 005380 --rl_method monkey --net monkey --start_date 20200101 --end_date 20211231
python main.py --mode test --ver v3 --name 035720a --stock_code 035720 --rl_method monkey --net monkey --start_date 20180101 --end_date 20191231
python main.py --mode test --ver v3 --name 035720b --stock_code 035720 --rl_method monkey --net monkey --start_date 20200101 --end_date 20211231
