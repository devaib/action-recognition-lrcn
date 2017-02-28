import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import cPickle as pickle
import os
import numpy as np
from collections import namedtuple

# load data
Batch = namedtuple('Batch', ['data'])
filepath = os.path.dirname(os.path.realpath(__file__))
inputvec_path = os.path.join(filepath, '../cache/test/inputvec')
name = 'test_jogging_person02_d1_1'
input_vec0 = pickle.load(open(os.path.join(inputvec_path, name + '.p'), 'rb'))
input_vec = input_vec0[0:1]

# load model
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50-kth', 20)
mod = mx.mod.Module(symbol=sym, context=mx.gpu())
mod.bind(for_training=False, data_shapes=[('data', (1, 20, 100, 100))])
mod.set_params(arg_params, aux_params)

# predict
mod.forward(Batch([mx.nd.array(input_vec)]))
prob = mod.get_outputs()[0].asnumpy()
prob = np.squeeze(prob)

a = np.argsort(prob)[::-1]
actionnames = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
for i in a[0:6]:
    print('probability=%f, class=%s' % (prob[i], actionnames[i]))

