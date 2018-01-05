#!/usr/bin/env python
import os
import pdb 
import numpy as np
import pickle
import gc
import time
from sklearn.decomposition import PCA
pca_dict={}
pca_dict['33']=PCA(n_components=1024)
pca_dict['36']=PCA(n_components=1024)
pca_dict['62']=PCA(n_components=1024)
base_path = '/data1/qtang/samsung/part_reid/train/Samsung/alidfcd_ivabox_sambox_partnet_baseline'

def concat(path1,path2,folder):
  path1_1=os.path.join(path1,'normed_feature')
  for weight in [4,10]:
    weight = float(weight)/10
    for file_name in os.listdir(path1_1):
      npz1=np.load(os.path.join(path1_1,file_name))
      npz2=np.load(os.path.join(path2,file_name))
      assert (npz1['img']==npz2['img']).all()
      imgs=npz1['img']
      a=np.sqrt((npz2['feat']**2).sum(axis=1))[:,None]
      feat_pca = pca_dict[folder].transform(npz2['feat']/a)
      feats=np.concatenate((npz1['feat'],feat_pca*weight),axis=1)
      assert len(feats[0])==1536
      path3=os.path.join(path1,'concat_sftmax_pca_1024/'+str(weight))
      if not os.path.exists(path3):
        os.makedirs(path3)
      with open(os.path.join(path3,file_name),'w') as f:
        np.savez(f,feat=feats,img=imgs)

def fit_concat(folder):
  path = base_path+'/forward_sftmax/feat_'+folder+'_sftmax/pool5_93'
  fit_data = np.load(path+'/1.npz')['feat']/np.sqrt((np.load(path+'/1.npz')['feat']**2).sum(axis=1))[:,None]
  for file_name in os.listdir(path):
    if file_name!='1.npz':
      npz=np.load(os.path.join(path,file_name))
      a=np.sqrt((npz['feat']**2).sum(axis=1))[:,None]
      fit_data=np.concatenate((fit_data,npz['feat']/a),axis=0)
      print 'processing '+file_name
  print 'concat_fit_data: '+folder+' done'
  return fit_data

print 'fiting 33...'
pca_dict['33'].fit(np.concatenate((fit_concat('query33'),fit_concat('evaLib33')),axis=0))
print 'fiting 36...'
pca_dict['36'].fit(np.concatenate((fit_concat('query36'),fit_concat('evaLib36')),axis=0))
print 'fiting 62...'
pca_dict['62'].fit(np.concatenate((fit_concat('query62'),fit_concat('evaLib62')),axis=0))
print 'fit --done'

for folder in ['query33','evaLib33','evalib33','evaLib36','query36','evalib36','evaLib62','evalib62','query62']:
  for iter in ['80000']:
    path1 = base_path+'/val/feat_'+folder+'_top40_qf_senet/iter'+iter+'/'
    path2 = base_path+'/forward_sftmax/feat_'+folder+'_sftmax/pool5_93'
    concat(path1,path2,folder[-2:])
  print 'transform '+folder+' --done'

