import pandas as pd
import numpy as np


def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'inst', 'frgn']
    chart_data['inst'] = pd.to_numeric(chart_data['inst'].str.replace(',', ''), errors='coerce')
    chart_data['frgn'] = pd.to_numeric(chart_data['frgn'].str.replace(',', ''), errors='coerce')
    return chart_data


def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
        prep_data['inst_ma{}'.format(window)] = prep_data['inst'].rolling(window).mean()
        prep_data['frgn_ma{}'.format(window)] = prep_data['frgn'].rolling(window).mean()
    return prep_data


def build_training_data(prep_data):
    training_data = prep_data

    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values
    training_data['inst_lastinst_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'inst_lastinst_ratio'] = \
        (training_data['inst'][1:].values - training_data['inst'][:-1].values) / \
        training_data['inst'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values
    training_data['frgn_lastfrgn_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'frgn_lastfrgn_ratio'] = \
        (training_data['frgn'][1:].values - training_data['frgn'][:-1].values) / \
        training_data['frgn'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]
        training_data['inst_ma%d_ratio' % window] = \
            (training_data['inst'] - training_data['inst_ma%d' % window]) / \
            training_data['inst_ma%d' % window]
        training_data['frgn_ma%d_ratio' % window] = \
            (training_data['frgn'] - training_data['frgn_ma%d' % window]) / \
            training_data['frgn_ma%d' % window]

    return training_data


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
