import threading
import abc
import numpy as np

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        
        inp = None
        if hasattr(self, 'num_steps'):
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)

        # 공유 신경망 사용
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
        else:
            self.head = self.shared_network
        
        # 공유 신경망 미사용
        # self.head = self.get_network_head(inp, self.output_dim)

        self.model = torch.nn.Sequential(self.head)
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()

    def predict(self, sample):
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(sample).float().to(device)
                pred = self.model(x).detach().cpu().numpy()
                pred = pred.flatten()
            return pred

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            self.model.train()
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_pred = self.model(x)
            _loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
            loss += _loss.item()
        return loss

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)
        elif net == 'lstm':
            return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_network_head((num_steps, input_dim), output_dim)

    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.load(model_path)
    
class DNN(Network):
    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Linear(inp[0], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)


class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)


class LSTMModule(torch.nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only

    def forward(self, x):
        output, (h_n, _) = super().forward(x)
        if self.use_last_only:
            return h_n[-1]
        return output


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 1, kernel_size),
            torch.nn.BatchNorm1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)
