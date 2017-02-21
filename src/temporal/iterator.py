import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import numpy as np
import random
from random import randint

class TemporalIter:
    def __init__(self, data_names, data_shapes, data,
                 label_names, label_shapes, label, batch_size=32):
        self.batch_size = batch_size
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self._size = data_shapes[0][0]
        self._cur_batch = 0
        self._current = 0
        self.data = None
        self.label = None
        self.datas = data
        self.labels = label

    def __iter__(self):
        return self

    def reset(self):
        self._cur_batch = 0
        self._current = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self._cur_batch < self.batch_size:
            self._cur_batch += 1
            self._current += self.batch_size
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex())
            return data_batch

        else:
            raise StopIteration

    def getpad(self):
        pad = self._cur_batch + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def getindex(self):
        return self._cur_batch

    def _get_batch(self):
        batch_list = random.sample(xrange(self._size), self.batch_size)
        self.data = [mx.nd.array(self.datas[batch_list])]
        self.label = [mx.nd.array(self.labels[batch_list])]