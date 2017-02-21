import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
import numpy as np
import random
from iterator import TemporalIter
from of_prep import stack_optical_flow

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

num_classes = 2
batch_per_gpu = 16
num_gpus = 1

sym = mx.sym.load('../model/resnet-50/resnet-50-symbol.json')
net = get_new_model(sym, num_classes)

# plot network
# from mxnet import visualization
# visualization.plot_network(sym)
# a = mx.viz.plot_network(sym, shape={"data":(4, 20, 224, 224)})
# a.render('resnet-50')

prefix = 'resnet-50'
batch_size = batch_per_gpu * num_gpus
#TODO: try Pickle
input_vec0, labels0 = stack_optical_flow('boxing', 0 ,224, 224)
# input_vec1, labels1 = stack_optical_flow('handwaving', 1 ,224, 224)
# input_vecs = np.append(input_vec0, input_vec1, axis=0)
# labels = np.append(labels0, labels1, axis=0)
input_vecs = input_vec0
labels = labels0

data_names = ['data']
data_shapes = [input_vecs.shape]
data = input_vecs
label_names = ['softmax_label']
label_shapes = [labels.shape]
label = labels
num_batches = 16

data = TemporalIter(data_names, data_shapes, data,
                    label_names, label_shapes, label,
                    num_batches)

mod = mx.mod.Module(symbol=net)
mod.fit(data, num_epoch=5)

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
