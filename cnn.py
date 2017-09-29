import tensorflow as tf
from collections import OrderedDict
from batch_constructor import Preprocess, Batch
from sklearn.model_selection import train_test_split
import sys
import numpy as np


class Net(object):
    LAYERS = []
    WEIGHTS = []

    def __init__(self, config):
        self.config = {
            'settings': self._settings,
            'input': self._input_layer,
            'convolution': self._convolution_layer,
            'dense': self._dense_layer}
        self.activations = {
            'sigmoid': tf.sigmoid,
            'relu': tf.nn.relu,
            'elu': tf.nn.elu,
            'tanh': tf.tanh,
            'drop': tf.nn.dropout,
            'linear': lambda x: x}
        self.layer_logs = {
            'convolution': 0,
            'dense': 0,
            'pooling': 0}
        self.optimizers = {
            'gradient': tf.train.GradientDescentOptimizer,
            'adadelta': tf.train.AdadeltaOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'momentum': tf.train.MomentumOptimizer,
            'adam': tf.train.AdamOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer}
        self.loss_dict = {
            'abs_dif': tf.losses.absolute_difference,
            'cos_dist': tf.losses.cosine_distance,
            'hinge': tf.losses.hinge_loss,
            'huber': tf.losses.huber_loss,
            'mean_sqrt': tf.losses.mean_squared_error}
        with tf.name_scope('Non_trainable_parameters'):
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(None, None),
                                        name="Input_data")
            self.labels = tf.placeholder(dtype=tf.float32,
                                         shape=(None, None),
                                         name='Training_labels')
        self._create_net(config)

    def _settings(self, optimizer='adam', loss_type='mean_sqrt', **kwargs):
        self.optimizer = self.optimizers[optimizer](**kwargs)
        self.loss = self.loss_dict[loss_type]

    def _weight_init(self, shape, name, stddev=0.1):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init, dtype=tf.float32, name=name)

    def _bias_init(self, amount, name, const=0.1):
        return tf.Variable(initial_value=[const for _ in range(
            amount)], dtype=tf.float32, name=name)

    def _input_layer(self, channels=None, window=None, ** kwargs):
        self.DATA = self.input
        self.LAYERS.append((self.DATA, (channels, window)))
        return 0

    def _convolution_layer(self,
                           filter_height=None, filter_width=None,
                           channels=None, strides=[1, 1, 1, 1],
                           activation='relu', padding='VALID', **kwargs):
        if len(strides) == 2:
            strides = [1, *strides, 1]
        data = self.LAYERS[-1][0]
        dims = self.LAYERS[-1][-1]
        if len(dims) == 2:
            data = tf.reshape(data, [-1, dims[0], dims[1], 1])
            in_chan = 1
        else:
            in_chan = dims[2]
        self.layer_logs['convolution'] += 1
        with tf.name_scope('Trainable_parameters_convolution'):
            name = "Convolution_" + str(self.layer_logs['convolution'])
            name_bias = "Convolution_" + \
                str(self.layer_logs['convolution']) + "_bias"
            init_bias = self._bias_init(amount=channels, name=name_bias)
            init = self._weight_init(
                [filter_height, filter_width, in_chan, channels], name)
            self.WEIGHTS.append((init, init_bias))
        if padding == 'VALID':
            n_hight = (dims[0] - filter_height) / strides[1] + 1
            n_width = (dims[1] - filter_width) / strides[2] + 1
        else:
            n_hight = dims[0] / strides[1]
            n_width = dims[1] / strides[2]
        self.LAYERS.append((self.activations[activation](tf.nn.conv2d(
            data, self.WEIGHTS[-1][0], strides=strides, padding=padding) +
            self.WEIGHTS[-1][1]), (n_hight, n_width, channels)))
        return 0

    def _dense_layer(self, output_neurons=None,
                     activation='linear', **kwargs):
        self.layer_logs['dense'] += 1
        data = self.LAYERS[-1][0]
        dims = self.LAYERS[-1][1]
        if len(dims) == 2:
            in_chan = dims[0] * dims[1]
        else:
            in_chan = tf.cast(dims[0] * dims[1] * dims[2], dtype=tf.int32)
            data = tf.reshape(data, [-1, in_chan])
        with tf.name_scope('Trainable_parameters_dense'):
            name = "Dense_" + str(self.layer_logs['dense'])
            name_bias = "Dense_" + str(self.layer_logs['dense']) + "_bias"
            init_bias = self._bias_init(
                amount=output_neurons, name=name_bias)
            init = self._weight_init(
                shape=(in_chan, output_neurons), name=name)
            self.WEIGHTS.append((init, init_bias))
            self.LAYERS.append((self.activations[activation](
                tf.matmul(data, self.WEIGHTS[-1][0]) + self.WEIGHTS[-1][1]), 1))
        return 0

    def _create_net(self, config):
        for key, value in config.items():
            self.config[key](**value)

    def train(self, num_classes=1):
        y = tf.reshape(self.labels, [-1, num_classes])
        loss = self.loss(y, self.LAYERS[-1][0])
        step = self.optimizer.minimize(loss)
        # step = tf.train.AdamOptimizer().minimize(loss)
        return loss, step


# class SimpleNet(object):
#     LAYERS = []
#     WEIGHTS = []

#     def __init__(self, config=[20, 10, 5, 1]):
#         self._parse_conf(config)

#     def _parse_conf(self, config):
#         self.data = tf.placeholder(
#             dtype=tf.float32, shape=(None, config[0]))
#         self.target = tf.placeholder(
#             dtype=tf.float32, shape=(None, config[-1]))
#         self.LAYERS.append(self.data)
#         self.WEIGHTS.append(tf.Variable(
#             tf.truncated_normal((config[0], config[1]), stddev=0.3), dtype=tf.float32))
#         c = 0
#         for layer in config[1:-1]:
#             self.WEIGHTS.append(tf.Variable(
#                 tf.truncated_normal((config[c], layer), stddev=0.3), dtype=tf.float32))
#             self.LAYERS.append(tf.nn.relu(
#                 tf.matmul(self.LAYERS[-1], self.WEIGHTS[-1])))
#             c += 1
#         self.WEIGHTS.append(tf.Variable(
#             tf.truncated_normal((config[-2], config[-1]), stddev=0.3), dtype=tf.float32))
#         self.LAYERS.append(tf.matmul(self.LAYERS[-1], self.WEIGHTS[-1]))

#     def train(self):
#         loss = tf.reduce_mean(tf.square(self.target - self.LAYERS[-1]))
#         step = tf.train.AdamOptimizer().minimize(loss)
#         return loss


if __name__ == '__main__':
    data = Preprocess('data1.csv')
    bal = data.fetch(n_cols=slice(0, 2),
                     n_rows=slice(1, -1), slicing=False)
    lea = data.fetch(n_cols=[16 + i for i in range(10)],
                     n_rows=slice(1, -1), slicing=False)
    data = Preprocess(bal, raw=True)
    bal = data.change()
    data = Preprocess(lea, raw=True)
    lea = data.change(altering='mean')
    Xtr, Xte, Ytr, Yte = train_test_split(
        bal, lea, train_size=0.8, random_state=13)
    batch = Batch(np.hstack((Xtr, Ytr)), batch_size=3)
    gen = batch.produce(100)

    conf = OrderedDict([
        ('input', {'channels': 2,
                   'window': 10}),
        ('convolution', {'filter_height': 1,
                         'filter_width': 3,
                         'channels': 5,
                         'activation': 'relu',
                         'drop': 1}),
        ('dense', {'output_neurons': 1}),
        ('settings', {'optimizer': 'gradient',
                      'learning_rate': 1e-3})
    ])
    a = Net(conf)
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        data = next(gen)
        data, target = data[:, :20], data[:, 20].reshape(-1, 1)
        try:
            for i in range(100):
                loss, _ = s.run(a.train(), feed_dict={
                             a.input: data, a.labels: target})
                if i % 10 == 0:
                    print(loss)
        except KeyboardInterrupt:
            print("Early interrupted. Final loss: {0:.2f}".format(loss))

    # net = SimpleNet()
    # with tf.Session() as s:
    #     s.run(tf.global_variables_initializer())
    #     for i in range(100):
    #         data = next(gen)
    #         err = s.run(net.train(), feed_dict={net.data: data[:, 0:20],
    #                                             net.target: data[:, 21].reshape(-1, 1)})
    #         print(err)
