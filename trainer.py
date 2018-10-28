from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import better_exceptions
except ImportError:
    pass

from six.moves import xrange

from util import log
from pprint import pprint

from model import Model
import tensorflow.contrib.slim as slim
from input_ops import create_input_ops

import os
import time
import numpy as np
import tensorflow as tf
import h5py

import sys
sys.path.append('./datasets/')

PATH_MODEL = './logs'

class Trainer(object):
    def __init__(self,
                 config,
                 dataset,
                 dataset_test):
        self.config = config
        hyper_parameter_str = config.model + '-is_' + str(config.img_size) + '-bs_' + str(config.batch_size) + \
        '-lr_' + "{:.2E}".format(config.learning_rate) + '-ur_' + str(config.update_rate)
        self.train_dir = PATH_MODEL+ '/%s-%s/train_dir/' % (
            hyper_parameter_str,
            time.strftime("%Y%m%d_%H%M%S")
        )
        self.test_dir = PATH_MODEL + '/%s-%s/test_dir/' % (
            hyper_parameter_str,
            time.strftime("%Y%m%d_%H%M%S")
        )

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        if not os.path.exists(self.test_dir): os.makedirs(self.test_dir)
        log.infov("Log Dir: %s", PATH_MODEL+ '/%s-%s/' % (
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
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.test_writer = tf.summary.FileWriter(self.test_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor =  tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
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
        pprint(self.batch_train)

        max_steps = 1000000

        output_save_step = 1000
        test_sample_step = 100

        for s in xrange(max_steps):
            step, summary, GAN_loss, d_loss, g_loss, s_loss, step_time, prediction_train, gt_train, g_img \
                = self.run_single_step(self.batch_train, step=s, is_train=True)

            # periodic inference
            if s % test_sample_step == 0:
                test_sum, prediction_test, gt_test, rf_precision, rf_recall, rf_f1, rf_acc = \
                    self.run_test(self.batch_test, is_train=False)
                self.test_writer.add_summary(test_sum, global_step=step)


            if s % 10 == 0:
                self.log_step_message(step, GAN_loss, d_loss, g_loss, s_loss, step_time)

            self.summary_writer.add_summary(summary, global_step=step)

            if s % output_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                save_path = self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)
                if self.config.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'g_img_'+str(s)+'.hy'), 'w')
                    f['image'] = g_img
                    f.close()

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.summary_op, self.model.GAN_loss, self.model.d_loss, self.model.g_loss,
                 self.model.S_loss, self.model.all_preds_real, self.model.all_targets, self.model.fake_img]

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
        [step, summary, GAN_loss, d_loss, g_loss, s_loss, all_preds, all_targets, g_img] = fetch_values[:9]

        _end_time = time.time()

        return step, summary, GAN_loss, d_loss, g_loss, s_loss,  (_end_time - _start_time), all_preds, all_targets, g_img

    def run_test(self, batch, is_train=False, repeat_times=8):

        batch_chunk = self.session.run(batch)

        fetch_gen = [self.summary_op, self.global_step, self.model.all_preds_real, self.model.all_targets]
        fetch_metrics_rf = [self.model.rf_precision, self.model.rf_recall, self.model.rf_f1, self.model.rf_acc]
        #fetch_metrics_au = [self.model.au_precision, self.model.au_recall, self.model.au_f1, self.model.au_acc]
        fetch = fetch_gen + fetch_metrics_rf

        [summary, step, all_preds, all_targets, rf_precision, rf_recall, rf_f1, rf_acc] = \
            self.session.run(fetch, feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False))

        return summary, all_preds, all_targets, rf_precision, rf_recall, rf_f1, rf_acc

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
    parser.add_argument('--dump_result', action='store_true', default=False)
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
