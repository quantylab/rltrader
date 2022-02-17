@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 카카오: 035720
@REM POSCO: 005490

for %%c in (005930 035720 005490) do (
    for %%n in (dnn lstm cnn) do (
        @REM train
        python main.py --mode train --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20180101 --end_date 20191231

        @REM test
        python main.py --mode test --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20200101 --end_date 20211231

        @REM predict
        python main.py --mode predict --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20211001 --end_date 20211031

        @REM update
        python main.py --mode update --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20200101 --end_date 20211231
    )
)

for %%n in (dnn lstm cnn) do (
    @REM train
    python main.py --mode train --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net %%n --start_date 20180101 --end_date 20191231

    @REM test
    python main.py --mode test --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net %%n --start_date 20200101 --end_date 20211231

    @REM predict
    python main.py --mode predict --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net %%n --start_date 20211001 --end_date 20211031

    @REM update
    python main.py --mode update --ver v3 --name all --stock_code 005930 035720 005490 --rl_method a3c --net %%n --start_date 20200101 --end_date 20211231
)
