from __future__ import division
from __future__ import print_function
import os
import tarfile
import subprocess
import argparse
import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
import glob
import sys
import progressbar
import git
import shutil
import configparser
import threading
from threading import Event, Thread

import facemotion

config = configparser.ConfigParser()
config.read('config.ini')
datasets_dir = config['PATH']['datasets_dir']
facemotion_dir = config['PATH']['facemotion_dir']

parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')

parser.add_argument('--model', type=str, choices=['AU', 'VA', 'BOTH'], default='BOTH')
parser.add_argument('--img_size', type=int, choices=[32, 64, 96], default=32)


def create_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.float32)

    print('Creating H5PY files...')

    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('-', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    name_data = args.model + '_' + str(args.img_size) + '_data.hy' 
    name_id = args.model + '_' + str(args.img_size) + '_id.txt'
    f = h5py.File(os.path.join(data_dir, name_data), 'w')
    data_id = open(os.path.join(data_dir, name_id), 'w')

    for i in range(image.shape[0]):
        bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')

        if shape:
            grp['image'] = np.reshape(image[i], shape)
        else:
            grp['image'] = image[i]
        label_vec = label[i]
        grp['label'] = label_vec.astype(np.float32)

    bar.finish()
    f.close()
    data_id.close()
    return

def check_h5py_file(h5py_dir):
    if os.path.exists(h5py_dir):
        name_data = args.model + '_' + str(args.img_size) + '_data.hy' 
        name_id = args.model + '_' + str(args.img_size) + '_id.txt'
        if os.path.isfile(os.path.join(h5py_dir + name_data)) and \
            os.path.isfile(os.path.join(h5py_dir + name_id)):
            return True
    else:
        os.mkdir(h5py_dir)
    return False

def process_dataset(download_path, img_size):

    data_dir = '/content/emotion-recognition-GAN/datasets/facemotion/final_images/01'
    # data_dir = os.path.join(download_path)

    h5py_dir = config['PATH']['h5py_dir']
    if check_h5py_file(h5py_dir):
        print('The H5PY files has already been created.')
        return
    else:
        print('The H5PY files are going to be created...')

    train_image = []
    test_image = []
    train_label = []
    test_label = []
    n_vas = 2
    n_aus = 8
    if args.model == 'AU':
        num_classes = n_aus
    elif args.model == 'VA':
        num_classes = n_vas
    elif args.model == 'BOTH':
        num_classes = n_vas + n_aus

    # list of videos part of the training data
    train_videos = facemotion.get_train_videos()
    # list of videos part of the testing data
    test_videos = facemotion.get_test_videos()

    # process image so that every image has the same size (cropping)
    def process_image(filename, img_size=img_size):
        image = Image.open(filename)
        width, height = image.size
        if (width > height):
            to_crop = width - height
            if (to_crop % 2 == 0):
                image = image.crop((to_crop / 2, 0, width - (to_crop / 2), height))
            else:
                image = image.crop((int(to_crop / 2), 0, width - (int(to_crop / 2)+1), height))
        elif (width < height):
            to_crop = height - width
            if (to_crop % 2 == 0):
                image = image.crop((0, to_crop / 2, width, height - (to_crop / 2)))
            else:
                image = image.crop((0, int(to_crop / 2), width, height - (int(to_crop/2)+1)))

        image = image.resize((img_size, img_size), Image.ANTIALIAS)
        im2array = np.array(image)
        if (str(im2array.shape) != str((img_size, img_size, 3))):
            print('The image has not been converted to the required size')
            print(im2array.shape)
            print(filename)
        return im2array

    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('*', '[', ']'), ' ',
                                           progressbar.Percentage()])
    print('Processing images...')
    bar.start()
    num_images_train = 0
    num_images_test = 0
    for counter, folder in enumerate(sorted(glob.glob(data_dir + 'final_images/*'))):
        bar.update((counter/64)*100)
        for img_filename in sorted(glob.glob(folder + '/*jpg')):
            if (int(os.path.basename(folder)) in train_videos):
                num_images_train += 1
                train_image.append(process_image(img_filename))
            elif (int(os.path.basename(folder)) in test_videos):
                num_images_test += 1
                test_image.append(process_image(img_filename))
    bar.finish()

    train_image = np.reshape(np.stack(train_image, axis=0), [num_images_train, img_size*img_size*3])
    test_image = np.reshape(np.stack(test_image, axis=0), [num_images_test, img_size*img_size*3])

    def create_labels(filename, labels):
        file = open(filename)
        lines = file.readlines()
        n_labels = len(lines)
        for i in range(n_labels):
            line = lines[i]
            line = line.split(' : ')[1]
            line_va = [line.split(" ")[0]] + [line.split(" ")[1]]
            line_au = line.split('[')[1].split(']')[0].split(' ')
            line_all = line_va + line_au
            if (args.model == 'AU'):
                labels.append(np.array(line_au, dtype=float))
            elif (args.model == 'VA'):
                labels.append(np.array(line_va, dtype=float))
            elif (args.model == 'BOTH'):
                labels.append(np.array(line_all, dtype=float))
        return n_labels

    num_labels_train = 0
    num_labels_test = 0
    for file in sorted(glob.glob(data_dir + '/final_annotations/*.txt')):
        file_name = os.path.basename(file)
        file_num = file_name.split('.')[0].split('_')[2]
        if (int(file_num) in train_videos):
            n_labels = create_labels(file, train_label)
            num_labels_train += n_labels
        elif (int(file_num) in test_videos):
            n_labels = create_labels(file, test_label)
            num_labels_test += n_labels

    assert num_images_train == num_labels_train, "Train images set and train labels set don't have the same size"
    assert num_images_test == num_labels_test, "Test images set and test labels set don't have the same size"

    print("Number of images in the training set : " + str(num_images_train))
    print("Number of images in the testing set : " + str(num_images_test))

    train_label = np.reshape(train_label, [num_labels_train, num_classes])
    test_label = np.reshape(test_label, [num_labels_test,num_classes])

    create_h5py(train_image, train_label, test_image, test_label, h5py_dir, [img_size, img_size, 3])


class MyThread(Thread):
    '''
    Created to see the progression of the dataset downloading
    '''
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not os.path.isdir(facemotion_dir + 'final_images/'):
            self.stopped.wait(0.1)
        print('bar')
        bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
        bar.start()
        folder_num = 1
        while not self.stopped.wait(0.5) and folder_num<64:
            empty = not os.listdir(facemotion_dir + 'final_images/'+str(folder_num).zfill(2))
            if not empty:
                bar.update((folder_num/64)*100)
                folder_num += 1
        bar.finish()

def downloaded_dataset():
    '''
    Check if the dataset has already been downloaded or not
    '''
    if os.path.isdir(facemotion_dir):
        ann_n = 0
        im_n = 0
        for annotation in sorted(glob.glob(facemotion_dir + 'final_annotations/*.txt')):
            ann_n += 1
            if not os.path.isfile(annotation):
                return False
        for folder in sorted(glob.glob(facemotion_dir + 'final_images/*')):
            im_n += 1
            if not os.listdir(folder):
                return False
        if (ann_n == 64 and im_n == 64):
            return True
        else:
            return False
    else:
        return False

def clone_repo():
    '''
    Clone the repository containing the original videos with annotations
    '''
    url_repo = config['PATH']['url_repo']
    if not os.path.exists(datasets_dir): os.mkdir(datasets_dir)
    try:
        print(downloaded_dataset)
        if not downloaded_dataset():
            if os.path.isdir(facemotion_dir):
                shutil.rmtree(facemotion_dir)
            print('The dataset is going to be downloaded from the repository...')
            stopFlag = Event()
            thread = MyThread(stopFlag)
            thread.start()
            git.Git(datasets_dir).clone(url_repo)
            stopFlag.set()
        else:
            print('The dataset has already been downloaded from the repository.')
    except KeyboardInterrupt:
        stopFlag.set()



if __name__ == '__main__':
    clone_repo()
    args = parser.parse_args()
    img_size = args.img_size
    process_dataset(facemotion_dir, img_size)