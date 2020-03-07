REM 현대차:005380 삼성전자:005930 NAVER:035420 한국전력:015760 LG화학:051910 셀트리온:068270

python main.py --stock_code 005380 --rl_method a2c --net lstm --num_steps 5 --output_name 005380 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
python main.py --stock_code 005930 --rl_method a2c --net lstm --num_steps 5 --output_name 005930 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
python main.py --stock_code 035420 --rl_method a2c --net lstm --num_steps 5 --output_name 035420 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
python main.py --stock_code 015760 --rl_method a2c --net lstm --num_steps 5 --output_name 015760 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
python main.py --stock_code 051910 --rl_method a2c --net lstm --num_steps 5 --output_name 051910 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
python main.py --stock_code 068270 --rl_method a2c --net lstm --num_steps 5 --output_name 068270 --learning --num_epoches 100 --lr 0.001 --start_epsilon 0.9
