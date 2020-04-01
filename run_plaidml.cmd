REM 현대차:005380 삼성전자:005930 한국전력:015760
REM NAVER:035420 LG화학:051910 셀트리온:068270

REM A2C
python main.py --stock_code 005380 --rl_method a2c --net lstm --num_steps 5 --output_name c_005380 --learning --num_epoches 1000 --lr 0.001 --start_epsilon 1 --discount_factor 0.9 --backend plaidml
python main.py --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name c_005930 --learning --num_epoches 1000 --lr 0.001 --start_epsilon 1 --discount_factor 0.9 --backend plaidml
python main.py --stock_code 015760 --rl_method a2c --net lstm --num_steps 5 --output_name c_015760 --learning --num_epoches 1000 --lr 0.001 --start_epsilon 1 --discount_factor 0.9 --backend plaidml

REM A3C
python main.py --stock_code 005380 005930 015760 --rl_method a3c --net lstm --num_steps 5 --output_name 005380_005930_015760 --learning --num_epoches 1000 --lr 0.001 --start_epsilon 1 --discount_factor 0.9 --backend plaidml

REM Testing
python main.py --stock_code 005380 --rl_method a2c --net lstm --num_steps 5 --output_name test_005380 --num_epoches 1 --start_epsilon 0 --start_date 20180101 --end_date 20181231 --reuse_models --value_network_name a2c_lstm_policy_b_005380 --policy_network_name a2c_lstm_value_b_005380 --backend plaidml


