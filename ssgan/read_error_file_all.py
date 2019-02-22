import argparse
import glob
import os
import pdb
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np

digits = re.compile(r'(\d+)')
_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
  return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)] 



def parse_all_files(config):

  results_path = '../logs/' + config.train_dir + '/train_dir/'

  f = []
  for dirpath,dirnames,_ in os.walk(results_path):
    f.extend(dirnames)
    f.sort(key=natural_sort_key)
    break

  # remove names other than model
  for n in f:
    if 'model' not in n:
      f.remove(n)

  AUS = []
  rf_tp = []
  rf_tn = []
  rf_fn = []
  rf_fp = []
  rf_precision = []
  rf_recall = []
  rf_acc = []
  rf_f1 = []
  ccc_v = []
  ccc_a = []
  mse_v = []
  mse_a = []
  mean_f1 = []
  mean_acc = []
  mean_f1_acc = []

  for line in f:
    if isfile(dirpath + '/' + str(line) + '/error.txt'):
      with open(dirpath + '/' + str(line) + '/error.txt') as fil:
        content = fil.readlines()
        if len(content) == 0:
          continue
    else:
      continue
  
    ###
    ### RF
    ###

    if content[0].rstrip().split(' ')[-1] == 'nan':
      rf_tp.append(-1.0)
    else:
      rf_tp.append(float(content[0].rstrip().split(' ')[-1]))
    
    if content[1].rstrip().split(' ')[-1] == 'nan':
      rf_tn.append(-1.0)
    else:
      rf_tn.append(float(content[1].rstrip().split(' ')[-1]))
    
    if content[2].rstrip().split(' ')[-1] == 'nan':
      rf_fn.append(-1.0)
    else:
      rf_fn.append(float(content[2].rstrip().split(' ')[-1]))

    if content[3].rstrip().split(' ')[-1] == 'nan':
      rf_fp.append(-1.0)
    else:
      rf_fp.append(float(content[3].rstrip().split(' ')[-1]))
    

    if content[4].rstrip().split(' ')[-1] == 'nan':
      rf_precision.append(-1.0)
    else:
      rf_precision.append(float(content[4].rstrip().split(' ')[-1]))
    
    if content[5].rstrip().split(' ')[-1] == 'nan':
      rf_recall.append(-1.0)
    else:
      rf_recall.append(float(content[5].rstrip().split(' ')[-1]))

    if content[6].rstrip().split(' ')[-1] == 'nan':
      rf_acc.append(-1.0)
    else:
      rf_acc.append(float(content[6].rstrip().split(' ')[-1]))

    if content[7].rstrip().split(' ')[-1] == 'nan':
      rf_f1.append(-1.0)
    else:
      rf_f1.append(float(content[7].rstrip().split(' ')[-1]))



    ###
    ### VA
    ###
    if config.model in ['BOTH', 'VA']:

      # CCC Valence
      if content[9].rstrip().split(' ')[0] == 'nan':
        ccc_v.append(-1.0)
      else:
        ccc_v.append(float(content[9].rstrip().split(' ')[0]))
    
      # CCC Arousal
      if content[9].rstrip().split(' ')[1] == 'nan':
        ccc_a.append(-1.0)
      else:
        ccc_a.append(float(content[9].rstrip().split(' ')[1]))
      
      # MSE Valence
      if content[11].rstrip().split(' ')[0] == 'nan':
        mse_v.append(-1.0)
      else:
        mse_v.append(float(content[11].rstrip().split(' ')[0]))
    
      # MSE Arousal
      if content[11].rstrip().split(' ')[1] == 'nan':
        mse_a.append(-1.0)
      else:
        mse_a.append(float(content[11].rstrip().split(' ')[1]))

    
    ###
    ### AU
    ###

    if config.model in ['BOTH', 'AU']:

      # AU recall precision accuracy f1
      aus = []
      for j in range(8):
        au = []
        for k in range(4):
          if content[j-10].rstrip().split(' ')[k] == 'nan':
            au.append(-1.0)
          else:  
            au.append(float(content[j-10].rstrip().split(' ')[k]))
        aus.append(au)
      AUS.append(aus)

      # Mean F1 Score
      if content[-2].rstrip().split(' ')[-1] == 'nan':
        mean_f1.append(-1.0)
      else:
        mean_f1.append(float(content[-2].rstrip().split(' ')[-1]))

      # Mean Accuracy
      if content[-2].rstrip().split(' ')[-2] == 'nan':
        mean_acc.append(-1.0)
      else:
        mean_acc.append(float(content[-2].rstrip().split(' ')[-2]))
      
      # Mean of the means between F1 and Accuracy
      if content[-1].rstrip().split(' ')[-1] == 'nan':
        mean_f1_acc.append(-1.0)
      else:
        mean_f1_acc.append(float(content[-1].rstrip().split(' ')[-1]))

  def get_best_results():
    file = open('../logs/' + config.train_dir + '/0_results.txt', 'w')

    file.write("Best RF precision : \n")
    file.write(str(rf_precision) + '\n')
    file.write(str(f[np.argmax(rf_precision)]) + '\n')
    file.write(str(np.max(rf_precision)) + '\n')
    file.write('\n')

    file.write("Best RF recall : \n")
    file.write(str(rf_recall) + '\n')
    file.write(str(f[np.argmax(rf_recall)]) + '\n')
    file.write(str(np.max(rf_recall)) + '\n')
    file.write('\n')

    file.write("Best RF accuracy : \n")
    file.write(str(rf_acc) + '\n')
    file.write(str(f[np.argmax(rf_acc)]) + '\n')
    file.write(str(np.max(rf_acc)) + '\n')
    file.write('\n')

    file.write("Best RF F1 : \n")
    file.write(str(rf_f1) + '\n')
    file.write(str(f[np.argmax(rf_f1)]) + '\n')
    file.write(str(np.max(rf_f1)) + '\n')
    file.write('\n')

    if config.model in ['BOTH', 'VA']:

      file.write("Best CCC Valence : \n")
      file.write(str(ccc_v) + '\n')
      file.write(str(f[np.argmax(ccc_v)]) + '\n')
      file.write(str(np.max(ccc_v)) + '\n')
      file.write('\n')
      
      file.write("Best CCC Arousal : \n")
      file.write(str(ccc_a) + '\n')
      file.write(str(f[np.argmax(ccc_a)]) + '\n')
      file.write(str(np.max(ccc_a)) + '\n')
      file.write('\n')

      file.write("Best MSE Valence : \n")
      file.write(str(mse_v) + '\n')
      file.write(str(f[np.argmax(mse_v)]) + '\n')
      file.write(str(np.max(mse_v)) + '\n')
      file.write('\n')

      file.write("Best MSE Arousal : \n")
      file.write(str(mse_a) + '\n')
      file.write(str(f[np.argmax(mse_a)]) + '\n')
      file.write(str(np.max(mse_a)) + '\n')
      file.write('\n')

    if config.model in ['BOTH', 'AU']:  

      file.write("Best Mean Accuracy : \n")
      file.write(str(mean_acc) + '\n')
      file.write(str(f[np.argmax(mean_acc)]) + '\n')
      file.write(str(np.max(mean_acc)) + '\n')
      file.write('\n')

      file.write("Best Mean Accuracy : \n")
      file.write(str(mean_acc) + '\n')
      file.write(str(f[np.argmax(mean_acc)]) + '\n')
      file.write(str(np.max(mean_acc)) + '\n')
      file.write('\n')

      file.write("Best Mean F1 : \n")
      file.write(str(mean_f1) + '\n')
      file.write(str(f[np.argmax(mean_f1)]) + '\n')
      file.write(str(np.max(mean_f1)) + '\n')
      file.write('\n')

      file.write("Best Mean between the mean of F1 and Acc : \n")
      file.write(str(mean_f1_acc) + '\n')
      file.write(str(f[np.argmax(mean_f1_acc)]) + '\n')
      file.write(str(np.max(mean_f1_acc)) + '\n')
      file.write('\n')
    
      def get_au(au, mes):
        l = []
        for t in range(len(AUS)):
          l.append(AUS[t][au][mes])
        return l, f[np.argmax(l)], np.max(l)
    
      for au_n, au in enumerate(['AU 1', 'AU 2', 'AU 4', 'AU 6', 'AU 12', 'AU 15', 'AU 20', 'AU 25']):
        for mes_n, mes in enumerate(['Recall', 'Precision', 'Accuracy', 'F1']):
          l, epoch, best = get_au(au_n,mes_n)
          file.write('Best ' + au + ' ' + mes + ': \n')
          file.write(str(l) + '\n')
          file.write(str(epoch) + '\n')
          file.write(str(best) + '\n')
          file.write('\n')

  

  get_best_results()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-td', '--train_dir', type=str)
  config = parser.parse_args()
  config.model = config.train_dir.split('-')[0]

  parse_all_files(config)


if __name__ == '__main__':
  main()
