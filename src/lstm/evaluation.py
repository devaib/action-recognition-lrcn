import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import cPickle as pickle
import os
import numpy as np
from collections import namedtuple
from img_io import ImageIter
from kth import KTH
from lrcn_model import LSTMInferenceModel
from lrcn import get_cnn, lstm_inference_symbol
import time

# hyperparameter
img_size = 227
batch_size = 4
seq_len = 20
buckets = [seq_len]
num_lstm_layer = 1
vocab_size = 128
num_embed = 256
num_hidden = 384
num_label = 6
ctx = mx.cpu()

# imdb
imdb = KTH('../cache/trainval')
data_name = 'data'
data_shape = (100, 100)
label_name = 'softmax_label'

# intput shapes
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

# load data
Batch = namedtuple('Batch', ['data'])
# filepath = os.path.dirname(os.path.realpath(__file__))
# inputvec_path = os.path.join(filepath, '../cache_imagestack/test/inputvec')
# name = 'test_handclapping_person08_d1_1_1'
# input_vec0 = pickle.load(open(os.path.join(inputvec_path, name + '.p'), 'rb'))
# input_vec = input_vec0[0:1]

imdb = KTH('../cache_imagestack/test')
data_names = ['data']
data_shapes = (100, 100)
label_names = ['softmax_label']
data_batch = ImageIter(imdb, seq_len, buckets, batch_size, data_shape,
                 init_states, data_name, label_name,
                 shuffle=True, is_train=True)
input_vec = data_batch._data['data']
labels = data_batch._label['softmax_label']
# input_vec = data.next().data[0]

# load model
sym, arg_params, aux_params = mx.model.load_checkpoint('lrcn', 20)
mod = mx.mod.Module(symbol=sym, context=ctx)

prev_sym = get_cnn(seq_len)
model = LSTMInferenceModel(prev_sym, num_lstm_layer, seq_len,
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=num_label, arg_params=arg_params, ctx=ctx, dropout=0.2)

# mod.bind(for_training=False, data_shapes=[('data', (batch_size, seq_len, 100, 100)),
#                                             ('l0_init_h', (batch_size, num_hidden)),
#                                             ('l0_init_c', (batch_size, num_hidden)),
#                                             ('softmax_label', (batch_size, seq_len))])
# mod.set_params(arg_params, aux_params)
#
# mod.forward(Batch([mx.nd.array(input_vec)]))
# prob = mod.get_outputs()[0].asnumpy()

prob = model.forward(input_vec, False)
print prob.shape
print prob

for pr in prob:
    p = pr
    prob = np.squeeze(p)

    a = np.argsort(p)[::-1]
    print a[0]
# for i in a[0]:
#     print('probability=%f' %(prob[i]))

# # predict
# count = 0
# error = 0
# # mod.forward(Batch([mx.nd.array(input_vec)]))
# for batch_inx in range(200):
#     if batch_inx % 10 == 0:
#         print('evaluating on batch {}'.format(batch_inx))
#     mod.forward(Batch([input_vec]))
#     prob = mod.get_outputs()[0].asnumpy()
#     prob = np.squeeze(prob)
#     outputs = np.argsort(prob)
#     predicts = np.empty(batch_size)
#     for inx, output in enumerate(outputs):
#         predicts[inx] = output[-1]
#     labels = labels.asnumpy()
#
#     for i in range(batch_size):
#         count += 1
#         if not predicts[i] == labels[i]:
#             error += 1
#
#     data_batch.next()
#     input_vec = data_batch._data['data']
#     labels = data_batch._label['softmax_label']
#
# print 'total: {}, error: {}, precision: {}'.format(int(count), int(error), float(count-error) / count)


