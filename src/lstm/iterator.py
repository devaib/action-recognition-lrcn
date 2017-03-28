import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import numpy as np

class TemporalIter(mx.io.DataIter):
    def __init__(self, lmdb, data_shape,
                 batch_size=128, shuffle=False, is_train=True):
        """

        Parameters
        ----------
        lmdb: Imdb
            image database
        data_shape: int or (int, int)
            inputvec width and height to be resized
        batch_size: int
            batch size
        shuffle: boolean
            whether to shuffle initial inputvec list, default False
        is_train: boolean
            whether in training phase, default True
        """
        self._lmdb = lmdb
        self._data_shape = data_shape
        self.batch_size = batch_size
        self._shuffle = shuffle

        self._cur_batch = 0
        self._current = 0
        self._size = lmdb.num_data
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self.is_train = is_train

        self._get_batch()

    def __iter__(self):
        return self

    def reset(self):
        self._cur_batch = 0
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

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
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                         label=self._label.values(),
                                         pad=self.getpad(),
                                         index=self.getindex())
            return data_batch

        else:
            raise StopIteration

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def getindex(self):
        return self._cur_batch

    def _get_batch(self):
        # padding = self.getpad()
        # if padding is 0:
        #     batch_list = range(self._current, self._current + self.batch_size)
        # else:
        #     batch_list = range(self._current, self._size)
        #     batch_list += range(0, padding)
        # self._data = {'data': mx.nd.array(self.datas[batch_list])}
        # if self.is_train:
        #     self._label = {'softmax_label': mx.nd.array(self.labels[batch_list])}
        # else:
        #     self._label = {'softmax_label': None}

        batch_inputvec = mx.nd.zeros((self.batch_size, 20, self._data_shape[0], self._data_shape[1]))
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # padding
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            name = self._lmdb.inputvec_path_from_index(index)
            inputvec_path = self._lmdb.inputvec_path_from_name(name)
            label_path = self._lmdb.label_path_from_name(name)
            inputvec = self._lmdb.load_inputvec(inputvec_path)
            label = self._lmdb.load_label(label_path)
            # TODO: data augmentation
            batch_inputvec[i] = inputvec[0]
            if self.is_train:
                batch_label.append(label)
        self._data = {'data': batch_inputvec}
        if self.is_train:
            self._label = {'softmax_label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'softmax_label': None}






