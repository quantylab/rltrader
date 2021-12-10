call C:\Users\USER\Anaconda3\Scripts\activate.bat

@REM 삼성전자: 005930
@REM 카카오: 035720
@REM LG화학: 051910
@REM 현대차: 005380
@REM 셀트리온: 068270
@REM POSCO: 005490

@REM train (201901~202012)
@REM 삼성전자: 005930
python main.py --ver v3 --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name 005930_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\005930_train_a2c_lstm\005930_train_a2c_lstm_value.h5 models\
copy output\005930_train_a2c_lstm\005930_train_a2c_lstm_policy.h5 models\
@REM 카카오: 035720
python main.py --ver v3 --stock_code 035720 --rl_method a2c --net lstm --num_steps 5 --output_name 035720_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\035720_train_a2c_lstm\035720_train_a2c_lstm_value.h5 models\
copy output\035720_train_a2c_lstm\035720_train_a2c_lstm_policy.h5 models\
@REM LG화학: 051910
python main.py --ver v3 --stock_code 051910 --rl_method a2c --net lstm --num_steps 5 --output_name 051910_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\051910_train_a2c_lstm\051910_train_a2c_lstm_value.h5 models\
copy output\051910_train_a2c_lstm\051910_train_a2c_lstm_policy.h5 models\
@REM 현대차: 005380
python main.py --ver v3 --stock_code 005380 --rl_method a2c --net lstm --num_steps 5 --output_name 005380_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\005380_train_a2c_lstm\005380_train_a2c_lstm_value.h5 models\
copy output\005380_train_a2c_lstm\005380_train_a2c_lstm_policy.h5 models\
@REM 셀트리온: 068270
python main.py --ver v3 --stock_code 068270 --rl_method a2c --net lstm --num_steps 5 --output_name 068270_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\068270_train_a2c_lstm\068270_train_a2c_lstm_value.h5 models\
copy output\068270_train_a2c_lstm\068270_train_a2c_lstm_policy.h5 models\
@REM POSCO: 005490
python main.py --ver v3 --stock_code 005490 --rl_method a2c --net lstm --num_steps 5 --output_name 005490_train --learning --num_epoches 50 --lr 0.001 --discount_factor 0.9 --start_epsilon 1 --start_date 20190101 --end_date 20201231
copy output\005490_train_a2c_lstm\005490_train_a2c_lstm_value.h5 models\
copy output\005490_train_a2c_lstm\005490_train_a2c_lstm_policy.h5 models\

@REM test (202101~202109)
@REM 삼성전자: 005930
python main.py --ver v3 --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name 005930_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 005930_train_a2c_lstm_value --policy_network_name 005930_train_a2c_lstm_policy
@REM 카카오: 035720
python main.py --ver v3 --stock_code 035720 --rl_method a2c --net lstm --num_steps 5 --output_name 035720_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 035720_train_a2c_lstm_value --policy_network_name 035720_train_a2c_lstm_policy
@REM LG화학: 051910
python main.py --ver v3 --stock_code 051910 --rl_method a2c --net lstm --num_steps 5 --output_name 051910_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 051910_train_a2c_lstm_value --policy_network_name 051910_train_a2c_lstm_policy
@REM 현대차: 005380
python main.py --ver v3 --stock_code 005380 --rl_method a2c --net lstm --num_steps 5 --output_name 005380_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 005380_train_a2c_lstm_value --policy_network_name 005380_train_a2c_lstm_policy
@REM 셀트리온: 068270
python main.py --ver v3 --stock_code 068270 --rl_method a2c --net lstm --num_steps 5 --output_name 068270_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 068270_train_a2c_lstm_value --policy_network_name 068270_train_a2c_lstm_policy
@REM POSCO: 005490
python main.py --ver v3 --stock_code 005490 --rl_method a2c --net lstm --num_steps 5 --output_name 005490_test --num_epoches 1 --start_epsilon 0 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 005490_train_a2c_lstm_value --policy_network_name 005490_train_a2c_lstm_policy

@REM predict (202110)
python predict.py --ver v3 --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --start_date 20211001 --end_date 20211031 --value_network_name 005930_train_a2c_lstm_value --policy_network_name 005930_train_a2c_lstm_policy

@REM update (202101~202109)
python main.py --ver v3 --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name 005930_update --num_epoches 50 --start_epsilon 1 --start_date 20210101 --end_date 20210930 --reuse_models --value_network_name 005930_train_a2c_lstm_value --policy_network_name 005930_train_a2c_lstm_policy --learning
