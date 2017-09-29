import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from LeapLog import get_log
log = get_log(__name__, 'WARNING')


class Batch(object):
    BATCHES = []

    def __init__(self, data, batch_size, random_state=None):
        self.data = data
        self.size = batch_size
        self.state = random_state
        self.n_batches = 0
        self.cur = 0
        self._iters_left = 0
        self._reshuffle()

    def __doc__(self):
        return (
            "Creating batches for TF-network with generator-like"
            " mechanism and seamless batch production combined"
            " with shuffling (with possibly fixed random state).\n"
            "Parameters:\n"
            "'data' - data as numpy array from which batches will creating.\n"
            "'batch_size' - size of batch.\n"
            "'random_state'[optional] - fixed random state for preserve"
            " batches structure in repeating training.\n"
            "Use example:\n"
            "batch = Batch(data, batch_size) #creating class instance"
            " with given data and batch size.\n"
            "sample = batch.produce(n_iters) #creating of generator for"
            " batch production with given number of required iterations.\n"
            "next(sample) #taking next sample from the generator.")

    def _create_batches(self):
        n_batches = self.data.shape[0] // self.size
        self.BATCHES = []
        for i in range(self.size, n_batches, self.size):
            self.BATCHES.append(self.data[i - self.size: i, :])
        self.n_batches = len(self.BATCHES)
        return

    def _reshuffle(self):
        self.data = shuffle(self.data, random_state=self.state)
        self._create_batches()
        return

    def _produce(self):
        while self.cur < self.n_batches:
            self.cur += 1
            self._iters_left -= 1
            yield self.BATCHES[self.cur - 1]

    def produce(self, num_iters=100):
        self._iters_left = num_iters
        while self._iters_left:
            gen = self._produce()
            try:
                yield next(gen)
            except StopIteration:
                log.debug("Current batches pool was ended. "
                          "Reshuffling things and continues.")
                self._reshuffle()
                self.cur = 0

    @property
    def info(self):
        print("Data shape is {0}".format(self.data.shape))
        print("There was created {0} batches with shape {1}".format(
            len(self.BATCHES), self.BATCHES[-1].shape))
        print(
            "Random state for data reshuffling was set to {0}".format(
                self.state))


class Preprocess(object):
    DATA = None

    def __init__(self, filename, raw=False, *args):
        if raw:
            self.DATA = filename
        else:
            self._load(filename)

    def _is_csv(self, filename):
        _, file_extension = os.path.splitext(filename)
        if file_extension == '.csv':
            return True
        elif file_extension:
            raise ValueError("Only .csv files are allowed.")
        return False

    def _load(self, filename):
        if self._is_csv(filename):
            self.DATA = np.genfromtxt(filename, delimiter=',')
            return 0
        else:
            to_return = []
            for file in glob.glob(os.path.join(filename, '*.csv')):
                to_return.append(np.genfromtxt(file, delimiter=','))
            if len(to_return):
                self.DATA = np.concatenate(to_return, axis=0)
                print(self.DATA.shape)
                return 0
            else:
                return 1

    def fetch(self, n_cols, n_rows=None, split=None,
              rnd_state=None, slicing=True):
        if n_rows and slicing:
            data = self.DATA[:n_rows, :n_cols]
        elif n_rows:
            data = self.DATA[n_rows, n_cols]
        elif slicing:
            data = self.DATA[:, :n_cols]
        else:
            data = self.DATA[:, n_cols]
        if split:
            data = train_test_split(
                data, train_size=split, random_state=rnd_state)
        return data

    def change(self, split=None, length=10, channels=None, altering='channel'):
        if split:
            if hasattr(split, '__len__'):
                data = self.DATA[:, split]
            else:
                data = self.DATA[:, :split]
        else:
            data = self.DATA
        if channels:
            data = data[:, channels]
        if altering == 'sample':
            # take all channels for 'length' times and combine them as a sample
            temp = np.hstack([i for i in data[:length, :]])
            for i in range(length, data.shape[0] // length - 1, length):
                temp = np.vstack(
                    (temp, np.hstack([j for j in data[i:i + length, :]])))
        elif altering == 'channel':
            # take 'length' samples for each channel (forming a windows)
            lim = data.shape[1]
            temp = np.hstack([data[j, i] for i in range(lim)
                              for j in range(length)])
            for l in range(length, data.shape[0] // length - 1, length):
                temp = np.vstack(
                    (temp, np.hstack([data[j + l, i] for i in range(lim)
                                      for j in range(length)])))
        elif altering == 'mean':
            # mean of 'length' samples ?
            temp = np.mean(data[:length, :], axis=0)
            for l in range(length, data.shape[0] // length - 1, length):
                temp = np.vstack(
                    (temp, np.mean(data[l:l + length, :], axis=0)))
        return temp

    @property
    def info(self):
        print("Data shape is {0}".format(self.DATA.shape))
