import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


def train(stock_code, data):
    # 기간 필터링
    training_data = data[(data['date'] >= '2016-01-01') &
                                  (data['date'] <= '2016-12-31')]
    training_data = training_data.dropna()
    # testing_data = data[(data['date'] >= '2016-01-01') &
    #                               (data['date'] <= '2016-12-31')]
    testing_data = data[(data['date'] >= '2017-01-01') &
                                  (data['date'] <= '2017-12-31')]
    testing_data = testing_data.dropna()

    # 차트 데이터 분리
    features = ['date', 'open', 'high', 'low', 'close', 'volume']
    training_chart_data = training_data[features]
    testing_chart_data = testing_data[features]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]
    testing_data = testing_data[features_training_data]

    # 강화학습 시작
    min_trading_unit = 1
    max_trading_unit = 1
    delayed_reward_threshold = .05
    start_epsilon = .5
    model_path = ''
    if stock_code == '005930':  # 삼성전자
        min_trading_unit = 1
        max_trading_unit = 1
        delayed_reward_threshold = .05
        model_path = os.path.join(settings.BASE_DIR, 'models/005930/model_20180318093401.h5')
    if stock_code == '000660':  # SK하이닉스
        min_trading_unit = 10
        max_trading_unit = 10
        delayed_reward_threshold = .05
        model_path = os.path.join(settings.BASE_DIR, 'models/000660/model_20180318105259.h5')
    if stock_code == '005380':  # 현대차
        min_trading_unit = 5
        max_trading_unit = 5
        delayed_reward_threshold = .02
        model_path = os.path.join(settings.BASE_DIR, 'models/005380/model_20180328005205.h5')
    if stock_code == '051910':  # LG화학
        min_trading_unit = 1
        max_trading_unit = 1
        delayed_reward_threshold = .05
        model_path = os.path.join(settings.BASE_DIR, 'models/051910/model_20180318020318.h5')
    if stock_code == '035420':  # NAVER
        min_trading_unit = 1
        max_trading_unit = 1
        delayed_reward_threshold = .05
        model_path = os.path.join(settings.BASE_DIR, 'models/035420/model_20180318143434.h5')
    if stock_code == '015760':  # 한국전력
        min_trading_unit = 10
        max_trading_unit = 10
        model_path = os.path.join(settings.BASE_DIR, 'models/015760/model_20180318032850.h5')
    if stock_code == '030200':  # KT
        min_trading_unit = 20
        max_trading_unit = 20
        model_path = os.path.join(settings.BASE_DIR, 'models/030200/model_20180318001555.h5')
    if stock_code == '035250':  # 강원랜드
        min_trading_unit = 30
        max_trading_unit = 30
        model_path = os.path.join(settings.BASE_DIR, 'models/035250/model_20180318043300.h5')
    if stock_code == '009240':  # 한샘 x
        min_trading_unit = 5
        max_trading_unit = 5
        model_path = os.path.join(settings.BASE_DIR, 'models/009240/model_20180318035122.h5')

    # 학습
    # policy_learner = PolicyLearner(
    #     stock_code=stock_code, chart_data=training_chart_data, training_data=training_data,
    #     min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
    #     delayed_reward_threshold=delayed_reward_threshold, lr=.0001)
    # policy_learner.fit(balance=10000000, num_epoches=1000,
    #                    discount_factor=0, start_epsilon=start_epsilon)
    #
    # # 정책 신경망을 파일로 저장
    # model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    # model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    # policy_learner.policy_network.save_model(model_path)

    # 테스팅
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=testing_chart_data, training_data=testing_data,
        min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit)
    policy_learner.trade(model_path, balance=10000000)


if __name__ == '__main__':
    list_stock_code = [
        '005930',  # 삼성전자 ok
        '000660',  # SK하이닉스 ok
        '005380',  # 현대차 ok
        '051910',  # LG화학 ok
        '035420',  # NAVER ok
        # '015760',  # 한국전력
        '030200',  # KT ok
        # '035250',  # 강원랜드
        # '009240',  # 한샘
    ]

    for stock_code in list_stock_code:
        # 로그 기록
        log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
        timestr = settings.get_time_str()
        file_handler = logging.FileHandler(filename=os.path.join(
            log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
        stream_handler = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)
        logging.basicConfig(format="%(message)s",
            handlers=[file_handler, stream_handler], level=logging.DEBUG)

        # 주식 데이터 준비
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         'chart_data/{}.csv'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)

        train(stock_code, training_data)
