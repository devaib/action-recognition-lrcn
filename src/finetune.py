import sys; sys.path.append('/home/binghao/workspace/mxnet/python')
import mxnet as mx
from mxnet import metric

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = './data/caltech-256/caltech-256-60-train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = './data/caltech-256/caltech-256-60-val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)


# sym, arg_params, aux_params = mx.model.load_checkpoint('model/resnet-50/resnet-50', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('model/full-resnet-152/full-resnet-152', 0)

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus, prefix):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(new_args, aux_params, allow_missing=True)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    mod.fit(train, val,
        num_epoch=8,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        epoch_end_callback=epoch_end_callback,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

num_classes = 256
batch_per_gpu = 32
num_gpus = 1

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

prefix = 'resnet-152'
batch_size = batch_per_gpu * num_gpus
(train, val) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus, prefix)