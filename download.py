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
sys.path.append('./datasets/')
import facemotion

parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')

parser.add_argument('--model', type=str, choices=['AU', 'VA', 'BOTH'], default='BOTH')
parser.add_argument('--img_size', type=int, choices=[32, 64, 96], default=32)


def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.float32)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    name_data = args.model + '_' + str(args.img_size) + '_data.hy' 
    name_id = args.model + '_' + str(args.img_size) + '_id.txt'
    f = h5py.File(os.path.join(data_dir, name_data), 'w')
    data_id = open(os.path.join(data_dir, name_id), 'w')
    for i in range(image.shape[0]):

        if i%(image.shape[0]/100)==0:
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

# def check_file(data_dir):
#     if os.path.exists(data_dir):
#         name_data = args.model + '_' + str(args.img_size) + '_data.hy' 
#         name_id = args.model + '_' + str(args.img_size) + '_id.txt'
#         if os.path.isfile(os.path.join(name_data)) and \
#             os.path.isfile(os.path.join(name_id)):
#             return True
#     else:
#         os.mkdir(data_dir)
#     return False

def download_facemotion(download_path, img_size):
    data_dir = os.path.join(download_path)

    # if check_file(data_dir):
    #     print('Facemotion was downloaded.')
    #     return

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

    num_images_train = 0
    num_images_test = 0
    for folder in sorted(glob.glob(data_dir + '/final_images/*')):
        for img_filename in sorted(glob.glob(folder + '/*jpg')):
            if (int(os.path.basename(folder)) in train_videos):
                num_images_train += 1
                train_image.append(process_image(img_filename))
            elif (int(os.path.basename(folder)) in test_videos):
                num_images_test += 1
                test_image.append(process_image(img_filename))

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

    dest_dir = './datasets/facemotion/'
    prepare_h5py(train_image, train_label, test_image, test_label, dest_dir, [img_size, img_size, 3])



if __name__ == '__main__':
    args = parser.parse_args()
    path = './datasets'
    if not os.path.exists(path): os.mkdir(path)

    PATH_TO_DATA = facemotion.get_data_dir()
    img_size = args.img_size
    #img_size = facemotion.get_data_info()[0]
    download_facemotion(PATH_TO_DATA, img_size)
