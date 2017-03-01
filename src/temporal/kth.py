import os
import numpy as np
import cPickle as pickle
from imdb import Imdb


class KTH(Imdb):
    def __init__(self, data_path, shuffle=False, is_train=False):
        """
        Implementation of Imdb for KTH dataset
        """
        super(KTH, self).__init__('KTH')
        self.data_path = data_path
        self.extension = '.p'
        self.is_train = is_train

        self.classes = ['boxing', 'handclapping', 'handwaving',
                        'jogging', 'running', 'walking']

        self.num_classes = len(self.classes)
        self.inputvec_index = self._load_inputvec_index(shuffle)
        self.num_data = len(self.inputvec_index)
        if self.is_train:
            self.labels = self._load_inputvec_labels()

    def _load_inputvec_index(self, shuffle):
        """
        Parameters
        ----------
        shuffle: boolean
            whether to shuffle the inputvec list

        Returns
        ----------
        entire list of inputvec
        """
        path = os.path.join(self.data_path, 'inputvec')
        inputvec = []
        for root, dirs, files in os.walk(path):
            for f in files:
                inputvec.append(f)
        if shuffle:
            np.random.shuffle(inputvec)
        return inputvec

    def inputvec_path_from_index(self, index):
        """
        given inputvec index, find out name
        Parameters
        ----------
        index: int
            index of specific inputvec
        Returns
        -------
        name of this inputvec(or label, actually the same)
        """
        assert self.inputvec_index is not None, "Dataset not initialized"
        name = self.inputvec_index[index]
        inputvec_file = os.path.join(self.data_path, 'inputvec', name)
        assert os.path.exists(inputvec_file), "Path does not exist: {}".format(inputvec_file)
        return name

    def inputvec_path_from_name(self, name):
        assert self.inputvec_index is not None, "Dataset not initialized"
        inputvec_file = os.path.join(self.data_path, 'inputvec', name)
        assert os.path.exists(inputvec_file), "Path does not exist: {}".format(inputvec_file)
        return inputvec_file

    def label_path_from_name(self, name):
        assert self.inputvec_index is not None, "Dataset not initialized"
        label_file = os.path.join(self.data_path, 'label', name)
        assert os.path.exists(label_file), "Path does not exist: {}".format(label_file)
        return label_file

    def load_inputvec(self, inputvec_path):
        """
        Returns
        -------
        ndarray of inputvec
        """
        input_vec = pickle.load(open(inputvec_path, 'rb'))
        return input_vec

    def load_label(self, label_path):
        label = pickle.load(open(label_path, 'rb'))
        return label

    def evaluate(self, detections):
        pass

# kth = KTH('../cache/trainval')
# name = kth.inputvec_path_from_index(1)
# inputvec_path = kth.inputvec_path_from_name(name)
# label_path = kth.label_path_from_name(name)
# inputvec = kth.load_inputvec(inputvec_path)
# label = kth.load_label(label_path)
