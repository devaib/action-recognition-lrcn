import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import numpy as np
import random
from random import randint

class TemporalIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data,
                 label_names, label_shapes, label, batch_size=16):
        self.batch_size = batch_size
        self._data_names = data_names
        self._label_names = label_names
        self._size = data_shapes[0][0]
        self._cur_batch = 0
        self._current = 0
        self._data = None
        self._label = None
        self.datas = data
        self.labels = label
        self._get_batch()

    def __iter__(self):
        return self

    def reset(self):
        self._cur_batch = 0
        self._current = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in self._label.items()]

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._cur_batch += 1
            self._current += self.batch_size
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(), label=self._label.values(), pad=self.getpad(), index=self.getindex())
            return data_batch

        else:
            raise StopIteration

    def getpad(self):
        pad = self._cur_batch + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def getindex(self):
        return self._cur_batch

    def _get_batch(self):
        # TODO: remove duplication
        batch_list = random.sample(xrange(self._size), self.batch_size)
        self._data = {'data': mx.nd.array(self.datas[batch_list])}
        self._label = {'softmax_label': mx.nd.array(self.labels[batch_list])}



