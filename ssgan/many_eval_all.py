import glob, os
import subprocess
import pdb

import re


digits = re.compile(r'(\d+)')
_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)] 

import sys
sys.path.append('/vol/gpudata/vjr17/ssgan_all_2/')


### just put here the folder where you have the weights
checkpoint_dir = '/vol/gpudata/vjr17/ssgan_all_2/logs/default-FACEMOTION_lr_0.0001_update_G2_D1-20180902-231441/train_dir'

def main():

  t=0
  if t==0:

      f = []
      for dirpath,dirnames,filenames in os.walk(checkpoint_dir):
        f.extend(filenames)
        f.sort(key=natural_sort_key)
        break  

      j = 1
      
      for ff in f:
        if ff[0:5]!='model':
          continue  

        if ff.split('/')[-1].split('.')[-1] == 'meta' and int(ff.split('-')[1].split('.')[0]) >= 1000*j   :  # if you want modify the 1000: here it takes every 1000 iteration


          num = ff.split('-')[1].split('.')[0]
          if os.path.exists(checkpoint_dir+'/model-'+str(num)):
            j+=1
            continue

        
          subprocess.call('CUDA_VISIBLE_DEVICES=0 python /vol/gpudata/vjr17/ssgan_all_2/evaler.py --dataset FACEMOTION --checkpoint_path='+checkpoint_dir+'/model-'+str(num),shell=True) 

          j += 1   

if __name__ == '__main__':
    main()
