import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
import pickle
from PIL import Image
import os
import gc

def

def stack_optical_flow(actionname, label, width, height):
    """
    Params:
        actionname: str
            name of action
        label: num
            index of label
        width: num
            resized image width
        height: num
            resized image height

    Return:
        input_vec:  ndarray(block x channel x height x width)
        labels:     ndarray(block)
    """
    channels = 20
    first_time = True
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        op_num = len(os.listdir(os.path.join(path, actionname, 'horizontal')))

        for op_start in range(op_num - channels):
            fx = []
            fy = []

            for i in range(op_start, op_start + channels):
                path_hort = os.path.join(actionname, 'horizontal', actionname + '_{}.jpg'.format(i))
                path_vert = os.path.join(actionname, 'vertical', actionname + '_{}.jpg'.format(i))
                imgh = Image.open(path_hort)
                imgv = Image.open(path_vert)
                imgh = imgh.resize((width, height))
                imgv = imgv.resize((width, height))
                fx.append(imgh)
                fy.append(imgv)

            flow_x = np.dstack((fx[0], fx[1], fx[2], fx[3], fx[4], fx[5], fx[6], fx[7], fx[8], fx[9]))
            flow_y = np.dstack((fy[0], fy[1], fy[2], fy[3], fy[4], fy[5], fy[6], fy[7], fy[8], fy[9]))
            inp = np.dstack((flow_x, flow_y))
            inp = np.expand_dims(inp, axis=0)

            if not first_time:
                input_vec = np.concatenate((input_vec, inp))
                labels = np.append(labels, label)
            else:
                input_vec = inp
                labels = np.array(label)
                first_time = False

        input_vec = np.rollaxis(input_vec, 3, 1)
        input_vec = input_vec.astype('float16', copy=False)
        labels = labels.astype('int', copy=False)
        gc.collect()

        return input_vec, labels

    except:
        return None, None


# input_vec, labels = stack_optical_flow('boxing', 0 ,150, 100)
#
# cv2.imshow('test', input_vec[0][0].astype('uint8'))
# cv2.imshow('test1', input_vec[0][1].astype('uint8'))
# cv2.waitKey()
