REM 현대차 삼성전자 NAVER 한국전력 LG화학 셀트리온
REM 005380 005930 035420 015760 051910 068270

REM python main.py --stock_code 015760 --rl_method a2c --net lstm --n_steps 5 --output_name 015760 --learning
python main.py --stock_code 005380 --rl_method a2c --net lstm --n_steps 5 --output_name 005380 --learning
python main.py --stock_code 005930 --rl_method a2c --net lstm --n_steps 5 --output_name 005930 --learning
python main.py --stock_code 035420 --rl_method a2c --net lstm --n_steps 5 --output_name 035420 --learning
python main.py --stock_code 051910 --rl_method a2c --net lstm --n_steps 5 --output_name 051910 --learning
python main.py --stock_code 068270 --rl_method a2c --net lstm --n_steps 5 --output_name 068270 --learning
