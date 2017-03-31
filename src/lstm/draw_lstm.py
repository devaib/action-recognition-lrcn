import sys; sys.path.append("/home/binghao/workspace/mxnet/python")
import mxnet as mx
import numpy as np
import pprint
from collections import namedtuple

pp = pprint.PrettyPrinter(indent=4)
batch_size = 4
seq_len = 20
num_lstm_layer = 1
vocab_size = 128
num_embed = 256
num_hidden = 384
num_label = 6

# CNN
data = mx.symbol.Variable(name="data")
input_data = mx.sym.SliceChannel(data=data, num_outputs=seq_len, name='sliced')

def get_symbol(bucket_num, input_data):
    # Alexnet
    # stage 1
    conv1 = mx.symbol.Convolution(name='conv'+str(bucket_num)+'_1', data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(name='conv'+str(bucket_num)+'_2', data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(name='conv'+str(bucket_num)+'_3', data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(name='conv'+str(bucket_num)+'_4', data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(name='conv'+str(bucket_num)+'_5', data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc = mx.symbol.FullyConnected(name='fullyconnected'+str(bucket_num)+'_1', data=flatten, num_hidden=4096)
    return fc

fcs = []
for i in range(seq_len):
    fcs.append(get_symbol(i, input_data[i]))

def concat_wrapper(args):
    return mx.symbol.Concat(*args, dim=1)
concat = concat_wrapper([fc for fc in fcs])

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
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
    return LSTMState(c=next_c, h=next_h)

def lstm_unroll(prev_sym, num_lstm_layer, seq_len,
                num_hidden, num_embed, num_label, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
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
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    label = mx.sym.Variable('softmax_label')
    sliced = mx.sym.SliceChannel(data=prev_sym, num_outputs=seq_len, squeeze_axis=1, name='sliced%d' % i)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = sliced[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    #label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    #label = [label_slice[t] for t in range(seq_len)]
    #label = mx.sym.Concat(*label, dim=0)
    #label = mx.sym.Reshape(data=label, target_shape=(0,))
    ################################################################################

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm

# intput shapes
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

data_shape = dict([('data', (batch_size, seq_len, 227, 227)),
                   ('softmax_label', (batch_size, seq_len))] + init_states)
prev_sym = concat
sym = lstm_unroll(prev_sym, num_lstm_layer, seq_len,
                  num_hidden=num_hidden, num_embed=num_embed,
                  num_label=num_label, dropout=0.2)

arg_names = sym.list_arguments()
out_names = sym.list_outputs()
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape)
# arguments
print 'arguments: '
pp.pprint(list(zip(arg_names, arg_shape)))

# output
print 'output: '
pp.pprint(list(zip(out_names, out_shape)))

plot = mx.viz.plot_network(sym,
                    shape={'data': (batch_size, seq_len, 227, 227),
                           'l0_init_h': (batch_size, num_hidden),
                           'l0_init_c': (batch_size, num_hidden),
                           'softmax_label': (batch_size, seq_len)},
                    node_attrs={"shape":'rect',"fixedsize":'false'})
plot.render('lrcn')

