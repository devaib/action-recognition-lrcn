from collections import namedtuple
import time
import math
import sys;
sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import numpy as np

# from pprint import pprint as print

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])


def lstm(num_hidden, indata, mask, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Unit symbol

    Parameters
    ----------
    num_hidden: int
        Hidden node in the LSTM unit

    in_data: mx.symbol
        Input data symbol

    prev_state: LSTMState
        Cell and hidden from previous LSTM unit

    param: LSTMParam
        Parameters of LSTM network

    seqidx: int
        The horizental index of the LSTM unit in the recurrent network

    layeridx: int
        The vertical index of the LSTM unit in the recurrent network

    dropout: float, optional in range (0, 1)
        Dropout rate on the hidden unit

    Returns
    -------
    ret: LSTMState
        Current LSTM unit state
    """
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    # dropout the hidden h
    if dropout > 0.:
        next_h = mx.sym.Dropout(next_h, p=dropout)
    # mask out the output
    # next_c = mx.sym.element_mask(next_c, mask, name="t%d_l%d_c" % (seqidx, layeridx))
    # next_h = mx.sym.element_mask(next_h, mask, name="t%d_l%d_h" % (seqidx, layeridx))
    # next_c = next_c * mask.reshape((mask.size))
    # next_h = next_h * mask.reshape((mask.size))
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, ignore_label=0, dropout=0.):
    """
    The unrolling function to provide a multi-layer LSTM network for a specify sequence length
    Parameters
    ----------
    num_lstm_layer: int
        number of lstm layers we will stack
    seq_len: int
        length of RNN we want to unroll
    input_size: int
        the input vocabulary size
    num_hidden: int
        number of hidden unit in a LSTM unit
    num_embed: int
        dimention of word embedding vector
    num_label: int
        target output label number
    ignore_label: int, optional
        which label should not be used for calculating loss
    dropout: float, optional
        dropout rate in LSTM

    Returns
    -------
    sm: mx.symbol
        An unrolled LSTM network
    """

    # For weight we will share over whole network, we use ```mx.sym.Variable``` to represent it
    embed_weight = mx.sym.Variable("embed_weight")  # embedding lookup table
    cls_weight = mx.sym.Variable("cls_weight")  # classifier weight
    cls_bias = mx.sym.Variable("cls_bias")  # classifier bias
    # Vertical initalization states and weights for LSTM unit
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)

    # Input data
    data = mx.sym.Variable('data')  # input data, shape (batch, seq_length)
    mask = mx.sym.Variable('mask')  # input mask, shape (batch, seq_length)
    label = mx.sym.Variable('softmax_label')  # labels, shape (batch, seq_length)
    # Embedding calculation
    # We take the input and get all the embedding once
    # Which means the output will be in shape (batch, seq_length, output_embedding_dim)
    # Then we slice it will ```seq_len``` output
    # Which means seq_len output symbol, each's output shape is (batch, output_embedding_dim)
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    maskvec = mx.sym.SliceChannel(data=mask, num_outputs=seq_len, squeeze_axis=1)

    # Now we can unroll the network
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]  # input to LSTM cell, comes from embedding

        # stack LSTM
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              mask=maskvec[seqidx],
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dropout)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        hidden_all.append(hidden)  # last output of stack LSTM units

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    # If we want to have attention, add it here.
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, ignore_label=ignore_label, name='softmax')

    return sm


def lstm_unroll_with_state(num_lstm_layer, seq_len, input_size,
                           num_hidden, num_embed, num_label, ignore_label=0, dropout=0.):
    """
    The unrolling function to provide a multi-layer LSTM network for a specify sequence length
    Parameters
    ----------
    num_lstm_layer: int
        number of lstm layers we will stack
    seq_len: int
        length of RNN we want to unroll
    input_size: int
        the input vocabulary size
    num_hidden: int
        number of hidden unit in a LSTM unit
    num_embed: int
        dimention of word embedding vector
    num_label: int
        target output label number
    ignore_label: int, optional
        which label should not be used for calculating loss
    dropout: float, optional
        dropout rate in LSTM

    Returns
    -------
    sm: mx.symbol
        An unrolled LSTM network
    """

    # For weight we will share over whole network, we use ```mx.sym.Variable``` to represent it
    embed_weight = mx.sym.Variable("embed_weight")  # embedding lookup table
    cls_weight = mx.sym.Variable("cls_weight")  # classifier weight
    cls_bias = mx.sym.Variable("cls_bias")  # classifier bias
    # Vertical initalization states and weights for LSTM unit
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)

    # Input data
    data = mx.sym.Variable('data')  # input data, shape (batch, seq_length)
    mask = mx.sym.Variable('mask')  # input mask, shape (batch, seq_length)
    label = mx.sym.Variable('softmax_label')  # labels, shape (batch, seq_length)
    # Embedding calculation
    # We take the input and get all the embedding once
    # Which means the output will be in shape (batch, seq_length, output_embedding_dim)
    # Then we slice it will ```seq_len``` output
    # Which means seq_len output symbol, each's output shape is (batch, output_embedding_dim)
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    maskvec = mx.sym.SliceChannel(data=mask, num_outputs=seq_len, squeeze_axis=1)

    # Now we can unroll the network
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]  # input to LSTM cell, comes from embedding

        # stack LSTM
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              mask=maskvec[seqidx],
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dropout)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        hidden_all.append(hidden)  # last output of stack LSTM units

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    # If we want to have attention, add it here.
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, ignore_label=ignore_label, name='softmax')

    outputs = [sm]
    # In the input we use init_c + init_h, so we will keep output in same convention
    for i in range(num_lstm_layer):
        state = last_states[i]
        outputs.append(mx.sym.BlockGrad(state.c, name="layer_%d_c" % i))  # stop back prop for last state
    for i in range(num_lstm_layer):
        state = last_states[i]
        outputs.append(mx.sym.BlockGrad(state.h, name="layer_%d_h" % i))  # stop back prop for last state
    return mx.sym.Group(outputs)


import collections


# simple batch is used for module to get data name, label name, data, label and bucket key
class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


# IO for bucketing LSTM
class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, buckets, vocab_size, batch_size, init_states):
        super(BucketSentenceIter, self).__init__()
        self.path = path
        self.buckets = sorted(buckets)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        # init
        self.data_name = ['data', 'mask']
        self.label_name = 'softmax_label'
        self._preprocess()
        self._build_vocab()
        sentences = self.content.split('<eos>')
        self.data = [[] for _ in self.buckets]
        self.mask = [[] for _ in self.buckets]
        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        discard_cnt = 0

        for sentence in sentences:
            sentence = self._text2id(sentence)
            bkt_idx = self._find_bucket(len(sentence))
            if bkt_idx == -1:
                discard_cnt += 1
                continue
            d, m = self._make_data(sentence, self.buckets[bkt_idx])
            self.data[bkt_idx].append(d)
            self.mask[bkt_idx].append(m)

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        mask = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                data[i_bucket][j, :] = self.data[i_bucket][j]
                mask[i_bucket][j, :] = self.mask[i_bucket][j]

        self.data = data
        self.mask = mask

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        print("Discard instance: %3d" % discard_cnt)
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key)),
                             ('mask', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

        self.reset()

    def _preprocess(self):
        self.content = open(self.path).read().lower().replace('\n', '<eos>')

    def _find_bucket(self, val):
        # lazy to use O(n) way
        for i, bkt in enumerate(self.buckets):
            if bkt > val:
                return i
        return -1

    def _make_data(self, sentence, bucket):
        # pad at the begining of the sequence
        mask = [1] * bucket
        data = [0] * bucket
        pad = bucket - len(sentence)
        data[pad:] = sentence
        mask[:pad] = [0 for i in range(pad)]
        return data, mask

    def _gen_bucket(self, sentence):
        # you can think about how to generate bucket candidtes in heuristic way
        # here we directly use manual defined buckets
        return self.buckets

    def _build_vocab(self):
        cnt = collections.Counter(self.content.split(' '))
        # take top k and abandon others as unknown
        # 0 is left for padding
        # last is left for unknown
        keys = cnt.most_common(self.vocab_size - 2)
        self.dic = {'PAD': 0}
        self.reverse_dic = {0: 'PAD', self.vocab_size - 1: "<UNK>"}  # is useful for inference from RNN
        for i in range(len(keys)):
            k = keys[i][0]
            v = i + 1
            self.dic[k] = v
            self.reverse_dic[v] = k
        print("Total tokens: %d, keep %d" % (len(cnt), self.vocab_size))

    def _text2id(self, sentence):
        sentence += " <eos>"
        words = sentence.split(' ')
        idx = [0] * len(words)
        for i in range(len(words)):
            if words[i] in self.dic:
                idx[i] = self.dic[words[i]]
            else:
                idx[i] = self.vocab_size - 1
        return idx

    def next(self):
        init_state_names = [x[0] for x in self.init_states]
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            init_state_names = [x[0] for x in self.init_states]
            data[:] = self.data[i_bucket][idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]

            label = self.label_buffer[i_bucket]
            label[:, :-1] = data[:, 1:]
            label[:, -1] = 0

            mask = self.mask_buffer[i_bucket]
            mask[:] = self.mask[i_bucket][idx]

            data_all = [mx.nd.array(data), mx.nd.array(mask)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data', 'mask'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

    __iter__ = next

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(int(len(self.data[i]) / self.batch_size))
            self.data[i] = self.data[i][:int(bucket_n_batches[i] * self.batch_size)]
        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        self.mask_buffer = []

        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            mask = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)
            self.mask_buffer.append(mask)

    def reset_states(self, states_data=None):
        if states_data == None:
            for arr in self.init_state_arrays:
                arr[:] = 0
        else:
            assert len(states_data) == len(self.init_state_arrays)
            for i in range(len(states_data)):
                states_data[i].copyto(self.init_state_arrays[i])

batch_size = 32
seq_len = 3
num_lstm_layer = 2
vocab_size = 128
num_embed = 256
num_hidden = 384

sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                  seq_len=seq_len,
                  input_size=vocab_size,
                  num_hidden=num_hidden,
                  num_embed=num_embed,
                  num_label=vocab_size,
                  ignore_label=0)

# intput shapes
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

data_shape = dict([('data', (batch_size, seq_len)),
                   # ('mask', (batch_size, seq_len)),
                   ('softmax_label', (batch_size, seq_len))] + init_states)

arg_names = sym.list_arguments()
out_names = sym.list_outputs()
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape)

# the argument of the unrolled network
print(list(zip(arg_names, arg_shape)))

# the output of the unrolled network
print(list(zip(out_names, out_shape)))

# new model
sym = lstm_unroll_with_state(num_lstm_layer=num_lstm_layer,
                             seq_len=seq_len,
                             input_size=vocab_size,
                             num_hidden=num_hidden,
                             num_embed=num_embed,
                             num_label=vocab_size,
                             ignore_label=0)

# intput shapes
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

data_shape = dict([('data', (batch_size, seq_len)),
                   # ('mask', (batch_size, seq_len)),
                   ('softmax_label', (batch_size, seq_len))] + init_states)

arg_names = sym.list_arguments()
out_names = sym.list_outputs()
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape)

# the argument of the unrolled network
print(list(zip(arg_names, arg_shape)))

# the output of the unrolled network
print(list(zip(out_names, out_shape)))

# params
num_lstm_layer = 2
num_hidden = 256
num_embed = 128
batch_size = 64

# state shape
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h
state_names = [x[0] for x in init_states]

# symbolic generate function
def sym_gen(seq_len):
    sym = lstm_unroll(num_lstm_layer, seq_len, len(vocab),
                      num_hidden=num_hidden, num_embed=num_embed,
                      num_label=len(vocab))
    data_names = ['data', 'mask'] + state_names
    label_names = ['softmax_label']
    return (sym, data_names, label_names)

# bucketing execution module
mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=[10,20,30,40,50], context=mx.cpu())

data_train = BucketSentenceIter(path="./book",
                                buckets=[10,20,30,40,50],
                                vocab_size=10000,
                                batch_size=batch_size,
                                init_states=init_states)

mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key, context=mx.gpu())

mod.fit(data_train, num_epoch=1,
        eval_metric='acc',
        batch_end_callback=mx.callback.Speedometer(batch_size, 50),
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.00001})














