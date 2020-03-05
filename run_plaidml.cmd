REM 현대차 삼성전자 NAVER 한국전력 LG화학 셀트리온
REM 005380 005930 035420 015760 051910 068270

python main.py --stock_code 005380 --rl_method dqn --net dnn --backend plaidml
python main.py --stock_code 005380 --rl_method dqn --net lstm --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method dqn --net cnn --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method pg --net dnn --backend plaidml
python main.py --stock_code 005380 --rl_method pg --net lstm --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method pg --net cnn --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method ac --net dnn --backend plaidml
python main.py --stock_code 005380 --rl_method ac --net lstm --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method ac --net cnn --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method a2c --net dnn --backend plaidml
python main.py --stock_code 005380 --rl_method a2c --net lstm --n_steps 5 --backend plaidml
python main.py --stock_code 005380 --rl_method a2c --net cnn --n_steps 5 --backend plaidml
