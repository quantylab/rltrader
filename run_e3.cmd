@echo off
setlocal enabledelayedexpansion

call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 카카오: 035720
@REM LG화학: 051910
@REM 현대차: 005380
@REM 셀트리온: 068270
@REM POSCO: 005490

@REM train (201901~202012)
for %%c in (005930 035720 051910 005380 068270 005490) do (
    for %%n in (dnn lstm cnn) do (
        set /A s=5
        if %%n == dnn (
            set /A s=1
        )

        python main.py --ver v3 --stock_code %%c --rl_method a2c --net %%n --num_steps !s! --output_name %%c_train --learning --num_epoches 100 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
        copy output\%%c_train_a2c_%%n\%%c_train_a2c_%%n_value.h5 models\
        copy output\%%c_train_a2c_%%n\%%c_train_a2c_%%n_policy.h5 models\

        python main.py --ver v3 --stock_code %%c --rl_method a2c --net %%n --num_steps !s! --output_name %%c_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210630 --reuse_models --value_network_name %%c_train_a2c_%%n_value --policy_network_name %%c_train_a2c_%%n_policy

        python predict.py --ver v3 --stock_code %%c --rl_method a2c --net %%n --num_steps !s! --start_date 20211001 --end_date 20211031 --value_network_name %%c_train_a2c_%%n_value --policy_network_name %%c_train_a2c_%%n_policy

        python main.py --ver v3 --stock_code %%c --rl_method a2c --net %%n --num_steps !s! --output_name %%c_update --num_epoches 50 --start_epsilon 1 --start_date 20210101 --end_date 20210630 --reuse_models --value_network_name %%c_train_a2c_%%n_value --policy_network_name %%c_train_a2c_%%n_policy --learning
    )
)
