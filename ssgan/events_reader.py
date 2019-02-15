from pathlib import Path

from PIL import Image

import tensorflow as tf
from ruamel.yaml import YAML

import argparse
import os

def read_event_file(event_file):
    '''
    Extracts all the images (real and fake) from the event file created for TensorBoard.
    '''

    dir = '/'.join(event_file.split('/')[:-1])
    if not os.path.exists(dir + '/images'):
        os.mkdir(dir + '/images')
    if not os.path.exists(dir + '/images/real'):
        os.mkdir(dir + '/images/real')
    if not os.path.exists(dir + '/images/fake'):
        os.mkdir(dir + '/images/fake')
    if not os.path.exists(dir + '/images/fake/0'):
        os.mkdir(dir + '/images/fake/0')
    if not os.path.exists(dir + '/images/fake/1'):
        os.mkdir(dir + '/images/fake/1')
    if not os.path.exists(dir + '/images/fake/2'):
        os.mkdir(dir + '/images/fake/2')

    real_c = 0
    fake_c_0 = 0
    fake_c_1 = 0
    fake_c_2 = 0
    for e in tf.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'img/real/image':
                real_c += 1
                f = open(dir + '/images/real/' + str(real_c) + '.png', 'w+')
                f.write(v.image.encoded_image_string)
                f.close()
            if v.tag == 'img/fake/image/0':
                fake_c_0 +=1
                f = open(dir + '/images/fake/0/' + str(fake_c_0) + '.png', 'w+')
                f.write(v.image.encoded_image_string)
                f.close()
            if v.tag == 'img/fake/image/1':
                fake_c_1 +=1
                f = open(dir + '/images/fake/1/' + str(fake_c_1) + '.png', 'w+')
                f.write(v.image.encoded_image_string)
                f.close()
            if v.tag == 'img/fake/image/2':
                fake_c_2 +=1
                f = open(dir + '/images/fake/2/' + str(fake_c_2) + '.png', 'w+')
                f.write(v.image.encoded_image_string)
                f.close()
    

if __name__ == '__main__':

    yaml_path = Path('config.yaml')
    yaml = YAML(typ='safe')
    config = yaml.load(yaml_path)
    paths = config['paths']

    parser = argparse.ArgumentParser()
    parser.add_argument('-ef', '--event_file', type=str, help='Path to the event file generated for TensorBoard')
    config = parser.parse_args()

    read_event_file(config.event_file)
