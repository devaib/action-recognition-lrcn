import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
from iterator import TemporalIter
import os
import numpy as np
import logging
import cPickle as pickle


# sym, arg_params, aux_params = mx.model.load_checkpoint('../model/resnet-50/resnet-50', 0)


def get_new_model(symbol, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    # new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return net

num_classes = 6
ctx = mx.gpu(0)

sym = mx.sym.load('../model/resnet-50/resnet-50-symbol.json')
net = get_new_model(sym, num_classes)

# plot network
# from mxnet import visualization
# visualization.plot_network(sym)
# a = mx.viz.plot_network(sym, shape={"data":(4, 20, 224, 224)})
# a.render('resnet-50')

# load data
filepath = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(filepath, '../cache/trainval')
inputvec_path = os.path.join(path, 'inputvec')
label_path = os.path.join(path, 'label')
input_vecs = None
labels = None


def fetch_trainval_data():
    setnames = ['train']
    actionnames = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    persons = ['person'+str(idx).zfill(2) for idx in range(11, 15)]
    conditions = ['d1', 'd2', 'd3', 'd4']
    subs = ['1', '2', '3', '4']
    filelist = []
    filepath = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(filepath, '../cache/trainval/inputvec')
    for setname in setnames:
        for actionname in actionnames:
            for person in persons:
                for condition in conditions:
                    for sub in subs:
                        filename = '_'.join([setname, actionname, person, condition, sub]) + '.p'
                        if os.path.exists(os.path.join(path, filename)):
                            filelist.append(filename)

    return filelist

namelist = fetch_trainval_data()
# TODO: rewrite iteration
for name in namelist:
    print 'processing ' + name
    input_vec0 = pickle.load(open(os.path.join(inputvec_path, name), 'rb'))
    label0 = pickle.load(open(os.path.join(label_path, name), 'rb'))
    if input_vecs is None or labels is None:
        input_vecs = input_vec0
        labels = label0
    else:
        input_vecs = np.append(input_vecs, input_vec0, axis=0)
        labels = np.append(labels, label0, axis=0)

data_names = ['data']
data_shapes = [input_vecs.shape]
data = input_vecs
label_names = ['softmax_label']
label_shapes = [labels.shape]
label = labels
batch_size = 32

data = TemporalIter(data_names, data_shapes, data,
                    label_names, label_shapes, label,
                    batch_size, shuffle=True)

logging.basicConfig(level=logging.INFO)
mod = mx.mod.Module(symbol=net, context=ctx)
model_prefix = 'resnet-50-kth'
checkpoint = mx.callback.do_checkpoint(model_prefix, 5)
mod.fit(data,
        num_epoch=20,
        batch_end_callback=mx.callback.Speedometer(batch_size, 100),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.1},
        eval_metric='acc',
        epoch_end_callback=checkpoint)

# train_sample_num = input_vecs.shape[0]
#
# ex = net.simple_bind(ctx=mx.gpu(), data=(batch_size, 20, 224, 224))
# args = dict(zip(net.list_arguments(), ex.arg_arrays))
#
# learning_rate = 0.1
# for iter in range(100):
#     batch_list = random.sample(xrange(train_sample_num), batch_size)
#     input_vec = input_vecs[batch_list]
#     label =labels[batch_list]
#     args['data'][:] = input_vec
#     args['softmax_label'][:] = label
#     ex.forward(is_train=True)
#     ex.backward()
#     for weight, grad in zip(ex.arg_arrays, ex.grad_arrays):
#         weight[:] -= learning_rate * (grad / batch_size)
#     if iter % 10 == 0:
#         acc = (mx.nd.argmax_channel(ex.outputs[0]).asnumpy() == label).sum()
#         final_acc = acc
#         print('iteration %d, accuracy %f' % (iter, float(acc)/label.shape[0]))
#
#
