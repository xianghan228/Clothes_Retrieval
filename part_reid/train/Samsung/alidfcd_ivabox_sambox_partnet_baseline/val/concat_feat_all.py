#!/usr/bin/env python
import os
import pdb 
import numpy as np
import pickle
import gc
import time

base_path = '/data1/qtang/samsung/part_reid/train/Samsung/alidfcd_ivabox_sambox_partnet_baseline'

def concat(path1,path2):
  path1_1=os.path.join(path1,'concat_feature/0.4')
  for weight in [5]:
    weight = float(weight)/10
    for file_name in os.listdir(path1_1):
      npz1=np.load(os.path.join(path1_1,file_name))
      npz2=np.load(os.path.join(path2,file_name))
      assert (npz1['img']==npz2['img']).all()
      imgs=npz1['img']
      a=np.sqrt((npz2['feat']**2).sum(axis=1))[:,None]
      feats=np.concatenate((npz1['feat'],npz2['feat']*weight/a),axis=1)
      path3=os.path.join(path1,'concat_feature_all/'+str(weight))
      if not os.path.exists(path3):
        os.makedirs(path3)
      with open(os.path.join(path3,file_name),'w') as f:
        np.savez(f,feat=feats,img=imgs)


for folder in ['evaLib36','query36','evalib36']:#,'evaLib62','evalib62','query62']:
  for iter in ['80000']:
    path1 = base_path+'/val/feat_'+folder+'_top40_qf_senet/iter'+iter+'/'
    path2 = base_path+'/forward_sftmax/feat_'+folder+'_sftmax/pool5_93'
    concat(path1,path2)


