from __future__ import absolute_import, division, print_function

import os
import pdb
import time
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np
from six.moves import xrange

import tensorflow as tf
from input_ops import check_data_id, create_input_ops
from model import Model
from ruamel.yaml import YAML
from util import log


class EvalManager(object):

    def __init__(self, config):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions_real = []
        self._predictions_false = []
        self._groundtruths = []

        self.config = config
        self.num_class = self.config.data_info[2]


    def add_batch(self, id, prediction_real, prediction_false, groundtruth):

        self._ids.append(id)
        self._predictions_real.append(prediction_real)
        self._predictions_false.append(prediction_false)
        self._groundtruths.append(groundtruth)

    def report(self, save_path):
        
        write_result = save_path+'/error.txt'
        write_prds = save_path+'/predictions.txt'
        write_labs = save_path+'/labels.txt'
        write_images = save_path+'/images.txt'

        n = self.num_class
        n_AU = 8 if n in (8, 10) else 0
        n_VA = 2 if n in (2, 10) else 0

        log.info("Computing scores...")

        predictions_real = np.reshape(self._predictions_real, (-1, n+1))
        predictions_false = np.reshape(self._predictions_false, (-1, n+1))
        labels = np.reshape(self._groundtruths, (-1, n))
        images = np.reshape(self._ids, (-1))

        with open(write_prds, "w") as aa:
            for line in predictions_real:  
                for li in line[:-1]: 
                    aa.write(str(li)+' ')
                aa.write('\t'+str(line[-1])+'\n')
        
        with open(write_labs, "w") as aa:
            for line in labels:
                for li in line[:-1]: 
                    aa.write(str(li)+' ')
                aa.write(str(line[-1])+'\n')
        
        with open(write_images, "w") as aa:
            for line in images:
                aa.write(str(line)+'\n')  


        ppredictions = []
        for elem in predictions_real[:,n_VA:]:
            tmp = []  
            for el in elem:  
                if el >= 0.5:
                    tmp.append(1)
                else:    
                    tmp.append(0)
            ppredictions.append(tmp)
        
        ppredictions =  np.reshape(ppredictions,[-1,n_AU+1])

        # predictions for false images
        fpredictions = []
        for elem in predictions_false[:,n_VA:]:
            tmp = []  
            for el in elem:  
                if el >= 0.5:
                    tmp.append(1)
                else:    
                    tmp.append(0)
            fpredictions.append(tmp)
        
        fpredictions =  np.reshape(fpredictions,[-1,n_AU+1])

        # real or fake predictions for real images
        real_fake_real = ppredictions[:,-1]
        # real or fakes predictions for fake images
        real_fake_fake = fpredictions[:,-1]
        tp = float(len([elem for elem in real_fake_real if elem == 0]))/len(real_fake_real)
        fn = float(len([elem for elem in real_fake_real if elem == 1]))/len(real_fake_real)
        tn = float(len([elem for elem in real_fake_fake if elem == 1]))/len(real_fake_fake)
        fp = float(len([elem for elem in real_fake_fake if elem == 0]))/len(real_fake_fake)
        rf_precision = tp / (tp+fp)
        rf_recall = tp / (tp+fn)
        rf_acc = (tp + tn) / (tp + tn + fp + fn)
        rf_f1 = 2*(rf_precision * rf_recall) / (rf_precision + rf_recall)

        with open(write_result, "w") as tr:
            tr.write('of the real images, the percentages classified as real is '+str(tp)+'\n')
            tr.write('of the false images, the percentages classified as false is '+str(tn)+'\n')
            tr.write('of the real images, the percentages classified as false is '+str(fn)+'\n')
            tr.write('of the false images, the percentages classified as real is '+str(fp)+'\n')
            tr.write('rf precision : {} \n'.format(str(rf_precision)))
            tr.write('rf recall : {} \n'.format(str(rf_recall)))
            tr.write('rf_acc : {} \n'.format(str(rf_acc)))
            tr.write('rf_f1 : {} \n'.format(str(rf_f1)))


        def concordance_cc2(r1, r2):
            mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
            return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)


        if self.config.model in ('VA', 'BOTH'):

            pred_v = predictions_real[:,0]
            pred_a = predictions_real[:,1]

            ccc_v = concordance_cc2(pred_v, labels[:,0])
            ccc_a = concordance_cc2(pred_a, labels[:,1])
         
            mse_v = ((pred_v - labels[:,0])**2).mean()
            mse_a = ((pred_a - labels[:,1])**2).mean()

            with open(write_result, "a") as tr:
                tr.write('ccc_v ccc_a'+'\n')
                tr.write(str(ccc_v) + ' ' + str(ccc_a)+ '\n')
                tr.write('mse_v mse_a'+'\n')
                tr.write(str(mse_v) + ' ' + str(mse_a) + '\n')


        if self.config.model in ('AU', 'BOTH'):

            recall_per_class = np.zeros((n_AU))
            f1_per_class = np.zeros((n_AU))
            accuracy_per_class = np.zeros((n_AU))
            precision_per_class = np.zeros((n_AU))

            for i in range(n_AU):
                ppr = ppredictions[:,i]
                llb = labels[:,i]
            
                tp=0
                tn=0
                fp=0
                fn=0
                total=0

                for pred_i, lab_i in zip(ppr,llb):
                    total += 1    
                
                    if lab_i == 1:  
                        if pred_i == 1:
                            tp+=1
                        else:
                            fn+=1  
                    elif lab_i == 0:  
                        if pred_i == 0:
                            tn+=1
                        else:
                            fp+=1  

                accuracy_per_class[i] = float((tp+tn))/ total
                ####### if at least for one AU we have recall=0 and/or precision=0  then i score f1 as -1
                try:
                    recall_per_class[i] = float(tp/ (tp+fn))
                except ZeroDivisionError:
                    recall_per_class[i] = -1
                try:
                    precision_per_class[i] = float(tp/ (tp+fp))
                except ZeroDivisionError:
                    precision_per_class[i] = -1

                if precision_per_class[i] == -1 or recall_per_class[i] == -1:
                    f1_per_class[i] = -1
                else:
                    if precision_per_class[i] == 0 and recall_per_class[i] == 0:
                        f1_per_class[i] = 0 
                    else:
                        f1_per_class[i] = 2 * precision_per_class[i] * recall_per_class[i] / float((precision_per_class[i] + recall_per_class[i]))

            with open(write_result, "a") as tr:
                tr.write('recall precision accuracy f1'+'\n')
                for el,el1,el2,el3 in zip(recall_per_class,precision_per_class,accuracy_per_class,f1_per_class):
                    tr.write(str(el)+' '+str(el1)+' '+str(el2)+' '+str(el3)+'\n')
                tr.write('mean accuracy f1 = '+str(np.mean(accuracy_per_class))+' '+str(np.mean(f1_per_class))+'\n')
                tr.write('total mean (accuracy+f1) = '+str(0.5*(np.mean(accuracy_per_class)+np.mean(f1_per_class)))+'\n')

        return 


class Evaler(object):
    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         num_threads=1,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=1000)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)


    def eval_run(self, config):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) + 1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager(config)
        try:
            for _ in xrange(max_steps):
                batch_chunk, prediction_pred_real, prediction_pred_fake, prediction_gt = \
                    self.run_single_step(self.batch)
                evaler.add_batch(batch_chunk['id'], prediction_pred_real, prediction_pred_fake, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        evaler.report(self.checkpoint_path)
        log.infov("Evaluation complete.")

    def run_single_step(self, batch, step=None, is_train=False):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, all_preds_real, all_preds_fake, all_targets, _] = self.session.run(
            [self.global_step, self.model.all_preds_real, self.model.all_preds_fake, self.model.all_targets, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return batch_chunk, all_preds_real, all_preds_fake, all_targets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['AU', 'VA', 'BOTH'])
    parser.add_argument('-is', '--img_size', type=int, choices=[32, 64, 96])
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-cp', '--checkpoint_path', type=str, help='The model to be evaluated')
    parser.add_argument('-td', '--train_dir', type=str, help='The last model saved will be evaluated')
    parser.add_argument('--data_id', nargs='*', default=None)
    config = parser.parse_args()

    import facemotion as dataset

    if config.checkpoint_path is not None:
        config.checkpoint_path = paths['logs_dir'] + config.checkpoint_path
        config.model = config.checkpoint_path.split('/')[-3].split('-')[0]
        config.img_size = int(config.checkpoint_path.split('/')[-3].split('-')[1].split('_')[1])
    elif config.train_dir is not None:
        config.train_dir = paths['logs_dir'] + config.train_dir
        config.model = config.train_dir.split('/')[-2].split('-')[0]
        config.img_size = int(config.train_dir.split('/')[-2].split('-')[1].split('_')[1])
    else:
        raise Exception('Precise the path where the model should be downloaded')

    config.dataset = 'FACEMOTION'
    config.data_info = dataset.get_data_info(config.img_size, config.model)
    config.conv_info = dataset.get_conv_info(config.img_size)
    config.deconv_info = dataset.get_deconv_info(config.img_size)
    _, dataset_test = dataset.create_default_splits(config.img_size, config.model)

    evaler = Evaler(config, dataset_test)
    evaler.eval_run(config)

if __name__ == '__main__':
    yaml_path = Path('config.yaml')
    yaml = YAML(typ='safe')
    config = yaml.load(yaml_path)
    paths = config['paths']
    main()
