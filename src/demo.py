import os, urllib
import mxnet as mx
import matplotlib
matplotlib.rc("savefig", dpi=100)
import matplotlib.pyplot as plt
import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
from collections import namedtuple

input_size = 224
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(dir_path, 'data', 'stopwatch.jpg')

with open('model/full-synset.txt', 'r') as f:
# with open('model/synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

sym, arg_params, aux_params = mx.model.load_checkpoint('model/full-resnet-152', 0)
# sym, arg_params, aux_params = mx.model.load_checkpoint('model/Inception-7', 1)


mod = mx.mod.Module(symbol=sym, context=mx.gpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,input_size,input_size))])
mod.set_params(arg_params, aux_params)

Batch = namedtuple('Batch', ['data'])

def get_image(url, show=True):
    filename = url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print('failed to download ' + url)
    if show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    return filename


def predict(filename, mod, synsets):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (input_size, input_size))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' % (prob[i], synsets[i]))

# url = 'http://thedigitalstory.com/2015/05/26/epson-p600-printer.jpg'
predict(image_path, mod, synsets)