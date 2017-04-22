import sys; sys.path.insert(0, "../../../python")
import numpy as np
import mxnet as mx

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.bucket_key = bucket_key
        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class ImageIter(mx.io.DataIter):
    def __init__(self, imdb, seq_len, buckets, batch_size, data_shape,
                 init_states, data_name='data', label_name='label',
                 shuffle=False, is_train=True):
        super(ImageIter, self).__init__()

        self.data_name = data_name
        self.label_name = label_name
        self.seq_len = seq_len

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]

        self.default_bucket_key = max(buckets)

        self.batch_size = batch_size
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        # from TemporalIter
        self._imdb = imdb
        self._data_shape = data_shape
        self.batch_size = batch_size
        self._shuffle = shuffle

        self._cur_batch = 0
        self._current = 0
        self._size = imdb.num_data
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
        return [(n, x.shape) for n, x in zip(self.data_names, self.data_all)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label_all)]

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._cur_batch += 1
            self._current += self.batch_size
            self._get_batch()
            # data_batch = mx.io.DataBatch(data=self.data_all,
            #                              label=self.label_all,
            #                              pad=self.getpad(),
            #                              index=self.getindex())
            data_batch = SimpleBatch(self.data_names, self.data_all,
                                     self.label_names, self.label_all,
                                     self.buckets[0])
            return data_batch

        else:
            raise StopIteration

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def getindex(self):
        return self._cur_batch

    def _get_batch(self):
        batch_inputvec = mx.nd.zeros((self.batch_size, self.seq_len, self._data_shape[0], self._data_shape[1]))
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
            name = self._imdb.inputvec_path_from_index(index)
            inputvec_path = self._imdb.inputvec_path_from_name(name)
            label_path = self._imdb.label_path_from_name(name)
            inputvec = self._imdb.load_inputvec(inputvec_path)
            label = self._imdb.load_label(label_path)
            # TODO: data augmentation
            batch_inputvec[i] = inputvec[0]
            if self.is_train:
                # TODO: stretch label for multiple outputs
                label = np.repeat(label, self.seq_len)
                batch_label.append(label)
        self._data = {self.data_name: batch_inputvec}
        if self.is_train:
            self._label = {self.label_name: np.array(batch_label)}
            init_state_names = [x[0] for x in self.init_states]
            # self.data_all = [mx.nd.array(self._data['data'])] + self.init_state_arrays
            self.data_all = [self._data[self.data_name]] + self.init_state_arrays
            self.label_all = [mx.nd.array(self._label[self.label_name])]
            self.data_names = [self.data_name] + init_state_names
            self.label_names = [self.label_name]
        else:
            self._label = {self.label_name: None}



