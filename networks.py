import os
import threading
import numpy as np


class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

def set_session(sess): pass


graph = DummyGraph()
sess = None


if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import set_session
    import tensorflow as tf
    graph = tf.get_default_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD


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
        self.model = None

    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x, y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lstm':
                return LSTMNetwork.get_network_head(
                    Input((num_steps, input_dim)))
            elif net == 'cnn':
                return CNN.get_network_head(
                    Input((1, num_steps, input_dim)))


class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation, 
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)
    

class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation, 
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = LSTM(256, dropout=0.1, 
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
            stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim, 1))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(128, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)
