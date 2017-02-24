import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
import cPickle as pickle
from PIL import Image
import os
import gc
import time


def prep_trainval_data():
    setnames = ['train', 'val', 'test']
    actionnames = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    persons = ['person'+str(idx).zfill(2) for idx in range(1, 26)]
    conditions = ['d1', 'd2', 'd3', 'd4']
    subs = ['1', '2', '3', '4']
    start_time = time.time()
    for setname in setnames:
        for actionname in actionnames:
            for person in persons:
                for condition in conditions:
                    for sub in subs:
                        if save_cache(setname, actionname, person, condition, sub):
                            elapsed_time = time.time() - start_time
                            m, s = divmod(elapsed_time, 60)
                            h, m = divmod(m, 60)
                            print "%d:%02d:%02d" % (h, m, s)


def save_cache(setname, actionname, person, condition, sub, path=None):
    """
    Params:
        setname: str
            'train', 'val' or 'test'
        actionname: str
            'boxing', 'handclapping', ...
        person: str
            'person01', 'person02'
        condition: str
            'd1', 'd2', 'd3', 'd4'
        sub: str
            '1', '2', '3', '4'

    Return:
        boolean
            if stacking optical flow fails
    """
    if path is None:
        path = '../data/human-action/optical-flow'
    cache_path = '../cache'
    filepath = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(filepath, path, setname)
    input_vec, label = stack_optical_flow(os.path.join(train_path, actionname, person, condition, sub), 0 ,150, 100)
    if input_vec is None:
        return False
    if setname is ('train' or 'val'):
        folder = 'trainval'
    elif setname is 'test':
        folder = 'test'
    pickle_name = '_'.join([setname, actionname, person, condition, sub])
    input_vec_dir = os.path.join(cache_path, folder, 'inputvec')
    if not os.path.exists(input_vec_dir):
        os.makedirs(input_vec_dir)
    label_dir = os.path.join(cache_path, folder, 'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    try:
        pickle.dump(input_vec, open(os.path.join(input_vec_dir, pickle_name + '.p'), 'wb'))
        pickle.dump(label, open(os.path.join(label_dir, pickle_name + '.p'), 'wb'))
        print 'successfully save to ' + os.path.join(input_vec_dir, pickle_name + '.p')
    except IOError:
        print 'failed to save to ' + os.path.join(input_vec_dir, pickle_name + '.p')
    return True


def compare(a, b):
    num0 = int(a.split('.')[0])
    num1 = int(b.split('.')[0])
    if num0 > num1:
        return 1
    elif num0 < num1:
        return -1
    else:
        return 0


def stack_optical_flow(path, label, width, height):
    """
    Params:
        path: str
            relative path to optical flows
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
        filepath = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(filepath, '..', path)
        batch_path = os.path.join(path, 'horizontal')
        op_num = len(os.listdir(batch_path))

        # traverse hort and vert folder
        for root, dirs, files in os.walk(path):
            if 'horizontal' in root:
                path_hort = root
                ofs = files    # horizontal optical flows
                ofs.sort(cmp=compare)
            if 'vertical' in root:
                path_vert = root

        start = int(ofs[0].split('.')[0])
        end = int(ofs[-1].split('.')[0])
        for op_start in range(start, end + 1 - channels):
            fx = []
            fy = []

            for i in range(op_start, op_start + channels):
                path_hort_img = os.path.join(path_hort, '{}.jpg'.format(i))
                path_vert_img = os.path.join(path_vert, '{}.jpg'.format(i))
                imgh = Image.open(path_hort_img)
                imgv = Image.open(path_vert_img)
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


# input_vec, labels = stack_optical_flow('data/human-action/optical-flow/train/jogging/person11/d1/1', 0 ,150, 100)
# cv2.imshow('test', input_vec[0][0].astype('uint8'))
# cv2.imshow('test1', input_vec[0][1].astype('uint8'))
# cv2.waitKey()

prep_trainval_data()
