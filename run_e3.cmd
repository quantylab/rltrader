@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 카카오: 035720
@REM LG화학: 051910
@REM 현대차: 005380
@REM 셀트리온: 068270
@REM POSCO: 005490

for %%c in (005930 035720 051910 005380 068270 005490) do (
    for %%n in (dnn lstm cnn) do (
        set /A s=5
        if %%n == dnn (
            set /A s=1
        )

        @REM train
        python main.py --action train --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20190101 --end_date 20201231

        @REM test
        python main.py --action test --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20210101 --end_date 20210630

        @REM predict
        python main.py --action predict --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20211001 --end_date 20211031

        @REM update
        python main.py --action update --ver v3 --name %%c --stock_code %%c --rl_method a2c --net %%n --start_date 20210101 --end_date 20210630
    )
)

