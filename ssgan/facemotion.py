from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import scipy.io
import scipy.ndimage as sn
import h5py
import glob
import configparser

from util import log

config = configparser.ConfigParser()
config.read('config.ini')
h5py_dir = config['PATH']['h5py_dir']

rs = np.random.RandomState(123)

class Dataset(object):

    def __init__(self, ids, img_size, model, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = model + '_' + str(img_size) + '_' + 'data.hy'

        file = os.path.join(h5py_dir, filename)
        log.info("Reading %s ...", file)

        try:
            self.data = h5py.File(file, 'r')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

    def get_data(self, id):
        # preprocessing and data augmentation
        m = self.data[id]['image'].value/255.
        l = self.data[id]['label'].value.astype(np.float32)

        # Data augmentation: rotate 0, 90, 180, 270
        """
        rot_num = np.floor(np.random.rand(1)*4)
        for i in range(rot_num):
            m = np.rot90(m, axes=(0, 1))
        m = m + np.random.randn(*m.shape) * 1e-2
        """
        return m, l

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )
        

def get_data_info(img_size, model):
    n_vas = 2
    n_aus = 8
    if model == 'AU':
        n_classes = n_aus
    elif model == 'VA':
        n_classes = n_vas
    elif model == 'BOTH':
        n_classes = n_aus + n_vas
    return np.array([img_size, img_size, n_classes, 3])

def get_conv_info(img_size):
    # IF img_size != 32 different return
    return np.array([64, 128, 256])

def get_deconv_info(img_size):
    # IF img_size != 32 different return !!
    return np.array([[384, 2, 1], [128, 4, 2], [64, 4, 2], [3, 6, 2]])
    #return np.array([[384, 2, 1], [128, 2, 2], [64, 4, 2], [32, 4, 2], [16, 4, 2], [3, 6, 2]])

def get_train_videos():
    return [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

def get_test_videos():
    return [1, 11, 16, 21, 29, 31, 39, 45, 49, 51]

def get_data_dir():
    PATH_TO_DATA = config['PATH']['facemotion_dir']
    return PATH_TO_DATA

def create_default_splits(img_size, model, is_train=True):
    ids = all_ids(img_size, model)
    n = len(ids)
    data_dir = get_data_dir()
    train_videos = get_train_videos()
    test_videos = get_test_videos()
    num_trains = 0
    num_tests = 0

    for folder in sorted(glob.glob(data_dir + '/final_images/*')):
        for _ in sorted(glob.glob(folder + '/*jpg')):
            if (int(os.path.basename(folder)) in train_videos):
                num_trains += 1
            elif (int(os.path.basename(folder)) in test_videos):
                num_tests += 1
    print("Size of the training set : " + str(num_trains))
    print("Size of the testing set : " + str(num_tests))

    dataset_train = Dataset(ids[:num_trains], img_size, model, name='train', is_train=False)
    dataset_test  = Dataset(ids[num_trains:], img_size, model, name='test', is_train=False)
    return dataset_train, dataset_test

def all_ids(img_size, model):

    id_filename = model + '_' + str(img_size) + '_id.txt'

    id_txt = os.path.join(h5py_dir, id_filename)
    print(id_txt)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

    return _ids
