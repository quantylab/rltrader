import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3'], default='v3')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c'])
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn'], default='dnn')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--start_date', default='20210101')
    parser.add_argument('--end_date', default='20211231')
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 로그 기록 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[stream_handler], level=logging.DEBUG)
    
    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))

    common_params = {}

    # 차트 데이터, 학습 데이터 준비
    chart_data, training_data = data_manager.load_data(args.stock_code, args.start_date, args.end_date, ver=args.ver)
    
    # 최소/최대 투자 단위 설정
    min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
    max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)

    # 공통 파라미터 설정
    common_params = {'rl_method': args.rl_method, 'net': args.net, 'num_steps': args.num_steps}

    # predict
    learner = None
    common_params.update({'stock_code': args.stock_code,
        'chart_data': chart_data, 
        'training_data': training_data,
        'min_trading_unit': min_trading_unit, 
        'max_trading_unit': max_trading_unit})
    if args.rl_method == 'dqn':
        learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
    elif args.rl_method == 'pg':
        learner = PolicyGradientLearner(**{**common_params, 'policy_network_path': policy_network_path})
    elif args.rl_method == 'ac':
        learner = ActorCriticLearner(**{**common_params, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
    elif args.rl_method == 'a2c':
        learner = A2CLearner(**{**common_params, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
    if learner is not None:
        print(learner.predict(balance=args.balance))
