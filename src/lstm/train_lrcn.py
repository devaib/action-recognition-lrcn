import sys; sys.path.append("/home/binghao/workspace/mxnet/python")
import mxnet as mx
import numpy as np
from lrcn import get_cnn, lstm_unroll
from kth import KTH
from img_io import ImageIter

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

img_size = 227
batch_size = 4
seq_len = 20
buckets = [seq_len]
num_lstm_layer = 1
vocab_size = 128
num_embed = 256
num_hidden = 384
num_label = 84
num_epoch = 2
learning_rate = 0.01
momemtum = 0.0
devs = mx.cpu()

prev_sym = get_cnn(seq_len)
sym = lstm_unroll(prev_sym, num_lstm_layer, seq_len,
                  num_hidden=num_hidden, num_embed=num_embed,
                  num_label=num_label, dropout=0.2)

# intput shapes
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

# iterator
imdb = KTH('../cache/trainval')
data_name = 'data'
data_shape = (100, 100)
label_name = 'softmax_label'

data = ImageIter(imdb, seq_len, buckets, batch_size, data_shape,
                 init_states, data_name, label_name,
                 shuffle=True, is_train=True)



