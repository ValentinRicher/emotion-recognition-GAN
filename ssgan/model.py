from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import conv2d, deconv2d, huber_loss
from util import log
import numpy as np

class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.num_class = self.config.data_info[2]
        self.c_dim = self.config.data_info[3]
        self.deconv_info = self.config.deconv_info
        self.conv_info = self.config.conv_info
        self.AUs = ["1","2","4","6","12","15","20","25"]

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.recon_weight = tf.placeholder_with_default(
            tf.cast(1.0, tf.float32), [])
        tf.summary.scalar("loss/recon_weight", self.recon_weight)

        self.build(is_train=is_train)


    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'], # [B, h, w, c]
            self.label: batch_chunk['label'], # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        # Weight annealing
        if step is not None:
            fd[self.recon_weight] = min(max(0, (1500 - step) / 1500), 1.0)*10
        return fd

    def build(self, is_train=True):

        n_VA = 2
        n_AU = 8
        n = self.num_class
        deconv_info = self.deconv_info
        conv_info = self.conv_info
        n_z = 100

        # Build loss {{{
        # =========
        def build_loss(d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image):
            #alpha = 0.9
            real_label = tf.concat([label, tf.zeros([self.batch_size, 1])], axis=1)
            #fake_label = tf.concat([(1-alpha)*tf.ones([self.batch_size, n])/n, alpha*tf.ones([self.batch_size, 1])], axis=1)
            fake_label = tf.concat([tf.zeros([self.batch_size, n]), tf.ones([self.batch_size, 1])], axis=1)		

            # Supervised loss
            s_loss = tf.reduce_mean(huber_loss(label[:,:n], d_real[:, :n]))


            # Discriminator/classifier loss {{
            
            d_loss_real_rf = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits[:,-1], labels=real_label[:,-1])
            d_loss_fake_rf = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits[:,-1], labels=fake_label[:,-1])

            if self.config.model in ('VA', 'BOTH'):
                d_loss_real_va = tf.losses.mean_squared_error(real_label[:,:n_VA], d_real_logits[:,:n_VA])
                d_loss_fake_va = tf.losses.mean_squared_error(fake_label[:,:n_VA], d_fake_logits[:,:n_VA])
                #d_loss_real_va = concordance_cc2(d_real_logits[:,:n_VA], real_label[:,:n_VA])
                #d_loss_fake_va = concordance_cc2(d_fake_logits[:,:n_VA], fake_label[:,:n_VA])

            if self.config.model in ('AU', 'BOTH'):
                d_loss_real_au = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits[:,n-n_AU:n], labels=real_label[:,n-n_AU:n])
                d_loss_fake_au = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits[:,n-n_AU:n], labels=fake_label[:,n-n_AU:n])


            if self.config.model == 'BOTH':
                d_loss_real = tf.reduce_mean(tf.reduce_mean(d_loss_real_va) + \
                 tf.reduce_mean(d_loss_real_au) + tf.reshape(d_loss_real_rf, [self.batch_size, 1]))
                d_loss_fake = tf.reduce_mean(tf.reduce_mean(d_loss_fake_va) + \
                 tf.reduce_mean(d_loss_fake_au) + tf.reshape(d_loss_fake_rf, [self.batch_size, 1]))

            elif self.config.model == 'AU':
                d_loss_real = tf.reduce_mean(d_loss_real_au + tf.reshape(d_loss_real_rf, [self.batch_size, 1]))
                d_loss_fake = tf.reduce_mean(d_loss_fake_au + tf.reshape(d_loss_fake_rf, [self.batch_size, 1]))

            elif self.config.model == 'VA':
                d_loss_real = tf.reduce_mean(tf.reduce_mean(d_loss_real_va) + \
                tf.reshape(d_loss_real_rf, [self.batch_size, 1]))
                d_loss_fake = tf.reduce_mean(tf.reduce_mean(d_loss_fake_va) + \
                tf.reshape(d_loss_fake_rf, [self.batch_size, 1]))


            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            # }}

            # Generator loss {{
            g_loss = tf.reduce_mean(tf.log(d_fake[:, -1]))
            # Weight annealing
            g_loss += tf.reduce_mean(huber_loss(real_image, fake_image)) * self.recon_weight
            # }}

            GAN_loss = tf.reduce_mean(d_loss + g_loss)

            return s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss



        # G takes ramdon noise and tries to generate images [B, h, w, c]
        def G(z, scope='Generator'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                z = tf.reshape(z, [self.batch_size, 1, 1, -1])
                g_1 = deconv2d(z, deconv_info[0], is_train, name='g_1_deconv')
                log.info('{} {}'.format(scope.name, g_1))
                g_2 = deconv2d(g_1, deconv_info[1], is_train, name='g_2_deconv')
                log.info('{} {}'.format(scope.name, g_2))
                g_3 = deconv2d(g_2, deconv_info[2], is_train, name='g_3_deconv')
                log.info('{} {}'.format(scope.name, g_3))
                g_4 = deconv2d(g_3, deconv_info[3], is_train, name='g_4_deconv', activation_fn=tf.tanh)
                log.info('{} {}'.format(scope.name, g_4))
                #g_5 = deconv2d(g_4, deconv_info[4], is_train, name='g_5_deconv')
                #log.info('{} {}'.format(scope.name, g_5))
                #g_6 = deconv2d(g_5, deconv_info[5], is_train, name='g_6_deconv', activation_fn=tf.tanh)
                #log.info('{} {}'.format(scope.name, g_6))
                output = g_4
                assert output.get_shape().as_list() == self.image.get_shape().as_list(), output.get_shape().as_list()
            return output

        # D takes images as input and tries to output class label [B, n+1]
        def D(img, scope='Discriminator', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                d_1 = conv2d(img, conv_info[0], is_train, name='d_1_conv')
                d_1 = slim.dropout(d_1, keep_prob=0.5, is_training=is_train, scope='d_1_conv/')
                if not reuse: log.info('{} {}'.format(scope.name, d_1))
                d_2 = conv2d(d_1, conv_info[1], is_train, name='d_2_conv')
                d_2 = slim.dropout(d_2, keep_prob=0.5, is_training=is_train, scope='d_2_conv/')
                if not reuse: log.info('{} {}'.format(scope.name, d_2))
                d_3 = conv2d(d_2, conv_info[2], is_train, name='d_3_conv')
                d_3 = slim.dropout(d_3, keep_prob=0.5, is_training=is_train, scope='d_3_conv/')
                if not reuse: log.info('{} {}'.format(scope.name, d_3))
                d_4 = slim.fully_connected(
                    tf.reshape(d_3, [self.batch_size, -1]), n+1, scope='d_4_fc', activation_fn=None)
                if not reuse: log.info('{} {}'.format(scope.name, d_4))
                output = d_4
                assert output.get_shape().as_list() == [self.batch_size, n+1]

                pred_rf = tf.reshape(tf.sigmoid(output[:,-1]), [self.batch_size,1])
                if self.config.model in ('VA', 'BOTH'):
                    pred_va = tf.nn.tanh(output[:, :n_VA])
                if self.config.model in ('AU', 'BOTH'):
                    pred_au = tf.sigmoid(output[:, n-n_AU:n])

                if self.config.model == 'BOTH':
                    pred = tf.concat([pred_va, pred_au, pred_rf], axis=1)
                elif self.config.model == 'AU':
                    pred = tf.concat([pred_au, pred_rf], axis=1)
                elif self.config.model == 'VA':
                    pred = tf.concat([pred_va, pred_rf], axis=1)
                
                return pred, output


        # Generator {{{
        # =========
        # input is a normal noise between -1 and 1
        z = tf.random_uniform([self.batch_size, n_z], minval=-1, maxval=1, dtype=tf.float32)
        fake_image = G(z)
        self.fake_img = fake_image
        # }}}

        # Discriminator {{{
        # =========
        d_real, d_real_logits = D(self.image, scope='Discriminator', reuse=False)
        d_fake, d_fake_logits = D(fake_image, scope='Discriminator', reuse=True)
        self.all_preds_real = d_real
        self.all_preds_fake = d_fake
        self.all_targets = self.label
        # }}}


        self.S_loss, d_loss_real, d_loss_fake, self.d_loss, self.g_loss, self.GAN_loss = \
            build_loss(d_real, d_real_logits, d_fake, d_fake_logits, self.label, self.image, self.fake_img)

        

        # Metrics {{{
        # =========

        # Metrics for Valence Arousal
        def concordance_cc2(predictions, labels):
            pred_mean, pred_var = tf.nn.moments(predictions, (0,))
            gt_mean, gt_var = tf.nn.moments(labels, (0,))
            mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))
            return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

        def build_metrics_va(label, d_real_logits):

            label_v = label[:,0]
            label_a = label[:,1]
            d_real_logits_v = d_real_logits[:,0]
            d_real_logits_a = d_real_logits[:,1]

            ccc_v = 1-concordance_cc2(d_real_logits_v, label_v)
            ccc_a = 1-concordance_cc2(d_real_logits_a, label_a)

            return ccc_v, ccc_a
        

        # Metrics for Action Units
        def build_metrics_au(label, d_real):

            label_au = label[:, n-n_AU:n]
            label_au = tf.cast(label_au, tf.float32)
            threshold = 0.5
            d_real_au = d_real[:, n-n_AU:n]
            pred = tf.cast(tf.greater(d_real_au, threshold*tf.ones([self.batch_size, 8])), tf.float32)

            ones = tf.cast(tf.ones([self.batch_size, 8]), tf.float32)
            tp = tf.count_nonzero(tf.multiply(label_au,pred), axis=0)
            fp = tf.count_nonzero(tf.multiply(tf.subtract(ones,label_au),pred), axis=0)
            tn = tf.count_nonzero(tf.multiply(tf.subtract(ones,label_au), tf.subtract(ones,pred)), axis=0)
            fn = tf.count_nonzero(tf.multiply(label_au, tf.subtract(ones,pred)), axis=0)

            au_precision = tp / (tp + fp)
            au_recall = tp / (tp + fn)
            au_f1 = 2*(au_precision*au_recall) / (au_precision+au_recall)
            au_acc = (tp + tn) / (tp + tn + fp + fn)

            return au_precision, au_recall, au_f1, au_acc

        def build_metrics_mean(au_precision, au_recall, au_f1, au_acc):
            au_precision_mean = tf.reduce_mean(au_precision)
            au_recall_mean = tf.reduce_mean(au_recall)
            au_f1_mean = tf.reduce_mean(au_f1)
            au_acc_mean = tf.reduce_mean(au_acc)
            return au_precision_mean, au_recall_mean, au_f1_mean, au_acc_mean


        # Metrics for distinguishing real and fake images
        def build_metrics_rf(d_real, d_fake):

            threshold = 0.5
            d_real_rf = d_real[:,-1]
            pred_real = tf.cast(tf.greater(d_real_rf, threshold*tf.ones([self.batch_size,1])), tf.float32)
            d_fake_rf = d_fake[:,-1]
            pred_fake = tf.cast(tf.greater(d_fake_rf, threshold*tf.ones([self.batch_size,1])), tf.float32)

            ones = tf.cast(tf.ones([self.batch_size, 1]), tf.float32)
            tp = tf.count_nonzero(tf.subtract(ones, pred_real))
            fn = tf.count_nonzero(pred_real)
            tn = tf.count_nonzero(pred_fake)
            fp = tf.count_nonzero(tf.subtract(ones, pred_fake))

            rf_precision = tp / (tp + fp)
            rf_recall = tp / (tp + fn)
            rf_f1 = 2*(rf_precision*rf_recall) / (rf_precision + rf_recall)
            rf_acc = (tp + tn) / (tp + tn + fp + fn)

            return rf_precision, rf_recall, rf_f1, rf_acc

        def build_metrics():

            tf.summary.scalar("loss/GAN_loss", self.GAN_loss)
            tf.summary.scalar("loss/S_loss", self.S_loss)
            tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
            tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
            tf.summary.scalar("loss/d_loss_fake", tf.reduce_mean(d_loss_fake))
            tf.summary.scalar("loss/g_loss", tf.reduce_mean(self.g_loss))
            tf.summary.image("img/fake", self.fake_img)
            tf.summary.image("img/real", self.image, max_outputs=1)
            #tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
            #tf.summary.image("label/pred_real", tf.reshape(tf.cast(tf.greater(d_real[:, n-n_AU:n], 0.5*tf.ones([self.batch_size, 8])), dtype=np.float32), [1, self.batch_size, n_AU, 1]))

            log.warn('\033[93mSuccessfully loaded the model.\033[0m')

            self.rf_precision, self.rf_recall, self.rf_f1, self.rf_acc = build_metrics_rf(d_real, d_fake)
            tf.summary.scalar("real_or_fake/precision", self.rf_precision)
            tf.summary.scalar("real_or_fake/recall", self.rf_recall)
            tf.summary.scalar("real_or_fake/f1", self.rf_f1)
            tf.summary.scalar("real_or_fake/accuracy", self.rf_acc)

            if self.config.model in ('VA', 'BOTH'):
                self.ccc_v, self.ccc_a = build_metrics_va(self.label, d_real_logits)
                tf.summary.scalar("concordance_correlation_coefficient/valence", self.ccc_v)
                tf.summary.scalar("concordance_correlation_coefficient/arousal", self.ccc_a)

            if self.config.model in ('AU', 'BOTH'):
                self.au_precision, self.au_recall, self.au_f1, self.au_acc = build_metrics_au(self.label, d_real)
                for i in range(n_AU):
                    au = "AU_" + self.AUs[i]
                    tf.summary.scalar("precision/" + au, self.au_precision[i])
                    tf.summary.scalar("recall/" + au, self.au_recall[i])
                    tf.summary.scalar("f1/" + au, self.au_f1[i])
                    tf.summary.scalar("accuracy/" + au, self.au_acc[i])
                self.au_precision_mean, self.au_recall_mean, self.au_f1_mean, self.au_acc_mean = build_metrics_mean(self.au_precision, self.au_recall, self.au_f1, self.au_acc)
                tf.summary("precision/au_mean", self.au_precision_mean)
                tf.summary("recall/au_mean", self.au_recall_mean)
                tf.summary("f1/au_mean", self.au_f1_mean)
                tf.summary("accuracy/au_mean", self.au_acc_mean)
        # }}}

        build_metrics()










