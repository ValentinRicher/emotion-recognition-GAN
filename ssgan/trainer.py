from __future__ import absolute_import, division, print_function

import configparser
import os
import time
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
from input_ops import create_input_ops
from model import Model
from ruamel.yaml import YAML
from util import log

try:
    import better_exceptions
except ImportError:
    pass


yaml_path = Path('config.yaml')
yaml = YAML(typ='safe')
config = yaml.load(yaml_path)
paths = config['paths']
h5py_dir = paths['h5py_dir']
logs_dir = paths['logs_dir']


class Trainer(object):

    def __init__(self,
                 config,
                 dataset,
                 dataset_test):

        self.config = config

        hyper_parameter_str = config.model + '-is_' + str(config.img_size) + '-bs_' + str(config.batch_size) + \
        '-lr_' + "{:.2E}".format(config.learning_rate) + '-ur_' + str(config.update_rate)
        self.train_dir = logs_dir+ '/%s-%s/train_dir/' % (
            hyper_parameter_str,
            time.strftime("%Y%m%d_%H%M%S")
        )
        self.test_dir = logs_dir + '/%s-%s/test_dir/' % (
            hyper_parameter_str,
            time.strftime("%Y%m%d_%H%M%S")
        )

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        if not os.path.exists(self.test_dir): os.makedirs(self.test_dir)
        log.infov("Log Dir: %s", logs_dir + '/%s-%s/' % (
            hyper_parameter_str,
            time.strftime("%Y%m%d_%H%M%S")
        ))

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(dataset, self.batch_size,
                                               is_training=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)

        # --- create model ---
        self.model = Model(config)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.check_op = tf.no_op()

        # --- checkpoint and monitoring ---
        all_vars = tf.trainable_variables()

        d_var = [v for v in all_vars if v.name.startswith('Discriminator')]
        log.warn("********* d_var ********** "); slim.model_analyzer.analyze_vars(d_var, print_info=True)

        g_var = [v for v in all_vars if v.name.startswith(('Generator'))]
        log.warn("********* g_var ********** "); slim.model_analyzer.analyze_vars(g_var, print_info=True)

        rem_var = (set(all_vars) - set(d_var) - set(g_var))
        print([v.name for v in rem_var]); assert not rem_var

        self.d_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.d_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate*0.5,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='d_optimize_loss',
            variables=d_var
        )

        self.g_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='g_optimize_loss',
            variables=g_var
        )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.train_dir)
        self.test_writer = tf.summary.FileWriter(self.test_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor =  tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.train_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")

        max_steps = self.config.max_steps
        test_sample_step = self.config.test_sample_step
        model_save_step = self.config.model_save_step
        summary_save_step = self.config.summary_save_step

        for s in xrange(max_steps):
            step, train_summary, GAN_loss, d_loss, g_loss, s_loss, step_time, g_img \
                = self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % 10 == 0:
                self.log_step_message(s, GAN_loss, d_loss, g_loss, s_loss, step_time)

            if (s % test_sample_step == 0) or (s % summary_save_step == 0):

                # periodic inference
                test_summary, GAN_loss, d_loss, g_loss, s_loss, step_time = \
                        self.run_test(self.batch_test, is_train=False)

                if s % test_sample_step == 0:
                    self.log_step_message(s, GAN_loss, d_loss, g_loss, s_loss, step_time, is_train=False)

                if s % summary_save_step == 0:
                    self.train_writer.add_summary(train_summary, global_step=step)
                    self.test_writer.add_summary(test_summary, global_step=step)

            if s % model_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)
                if self.config.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'g_img_'+str(s)+'.hy'), 'w')
                    f['image'] = g_img
                    f.close()


    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.summary_op, self.model.GAN_loss, self.model.d_loss, self.model.g_loss,
                 self.model.S_loss, self.model.fake_img]

        fetch.append(self.check_op)

        if step%(self.config.update_rate+1) > 0:
        # Train the generator
            fetch.append(self.g_optimizer)
        else:
        # Train the discriminator
            fetch.append(self.d_optimizer)

        fetch_values = self.session.run(fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, summary, GAN_loss, d_loss, g_loss, s_loss, g_img] = fetch_values[:7]

        _end_time = time.time()

        return step, summary, GAN_loss, d_loss, g_loss, s_loss,  (_end_time - _start_time), g_img


    def run_test(self, batch, is_train=False, repeat_times=8):

        _start_time = time.time()
        batch_chunk = self.session.run(batch)

        fetch_gen = [self.summary_op, self.global_step, self.model.GAN_loss, self.model.d_loss, self.model.g_loss,
                 self.model.S_loss]
        # fetch_metrics_rf = [self.model.rf_precision, self.model.rf_recall, self.model.rf_f1, self.model.rf_acc]
        # fetch_metrics_au = [self.model.au_precision, self.model.au_recall, self.model.au_f1, self.model.au_acc]
        fetch = fetch_gen

        [summary, _, GAN_loss, d_loss, g_loss, s_loss] = \
            self.session.run(fetch, feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False))
        
        _end_time = time.time()

        return summary, GAN_loss, d_loss, g_loss, s_loss, (_end_time - _start_time)


    def log_step_message(self, step, GAN_loss, d_loss, g_loss, s_loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "GAN loss: {GAN_loss:.5f} " +
                "Supervised loss: {s_loss:.5f} " +
                "D loss: {d_loss:.5f} " +
                "G loss: {g_loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step = step,
                         GAN_loss = GAN_loss,
                         d_loss = d_loss,
                         g_loss = g_loss,
                         s_loss = s_loss,
                         sec_per_batch = step_time,
                         instance_per_sec = self.batch_size / step_time
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['AU', 'VA', 'BOTH'], default='BOTH')
    parser.add_argument('-is', '--img_size', type=int, choices=[32, 64, 96], default=32)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-ur', '--update_rate', type=int, default=5)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum number of iterations')
    parser.add_argument('--model_save_step', type=int, default=1000, help='Frequency of model saving')
    parser.add_argument('--test_sample_step', type=int, default=100, help='Frequency of testing on the testing set')
    parser.add_argument('--summary_save_step', type=int, default=1000, help='Frequency of saving the elements to the TF summary')
    parser.add_argument('--dump_result', action='store_true', help='If the images are saved')
    parser.add_argument('--checkpoint', type=str, default=None)
    config = parser.parse_args()

    import facemotion as dataset

    config.dataset = 'FACEMOTION'
    config.data_info = dataset.get_data_info(config.img_size, config.model)
    config.conv_info = dataset.get_conv_info(config.img_size)
    config.deconv_info = dataset.get_deconv_info(config.img_size)
    dataset_train, dataset_test = dataset.create_default_splits(config.img_size, config.model)

    log.infov("PARAMETERS OF THE MODEL")
    log.infov("model: %s, image_size: %s, batch_size: %s, learning_rate: %.2E, update_rate: %s", \
     config.model, config.img_size, config.batch_size, config.learning_rate, config.update_rate)

    trainer = Trainer(config, dataset_train, dataset_test)
    trainer.train()

if __name__ == '__main__':
    main()
