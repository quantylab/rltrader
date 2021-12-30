import os

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    from quantylab.rltrader.networks.networks_pytorch import Network, DNN, LSTMNetwork, CNN
else:
    from quantylab.rltrader.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
