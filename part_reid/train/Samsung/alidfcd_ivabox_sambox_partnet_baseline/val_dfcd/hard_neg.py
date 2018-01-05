#!/usr/bin/env python
import os
import pickle
import numpy as np
import pdb
base_path = '/data1/qtang/samsung/part_reid/train/Samsung/alidfcd_ivabox_sambox_partnet_baseline/val_dfcd/result/pkl'
sub_paths  = ['cd/cd_Lib','df/df_Lib']
q_neg = {}

train_data = {}
with open('../train_list.txt','r') as f:
  for line in f.readlines():
    train_data[line.split()[0].split('/')[-1][:-4]] = int(line.split()[1])
  

def find_q_id(sub_path='',query=''):
  f=open('./query_gt_'+sub_path.split('/')[0]+'.txt','r')
  for line in f.readlines():
    if query in line:
      gts = line.split(',')[1].split(';')
      for gt in gts:
        if gt in train_data:
          label = train_data[gt]
          return label

def find_an_path(sub_path,an):
  f = open('./evaLib.txt','r')
  for line in f.readlines():
    if an+'.jpg' in line:
      return line[:-1]
    



for sub_path in sub_paths:
  path = os.path.join(base_path,sub_path)
  for file_name in os.listdir(path):
    if file_name.endswith('.pkl'):
      pkl_dic = pickle.load(open(os.path.join(path,file_name),'r'))
      query   = pkl_dic['retrieved_img']
      label   = int(query.split()[-1])
      ans     = np.array(pkl_dic['ans_img'])
      idx     = list(np.where(np.array(pkl_dic['in_or_not'])==0)[0])
      if len(idx)>1:
        w_ans   = list(ans[idx])[1:]
        q_neg[label] = w_ans
        print 'one query done'
        print str(label)+':'+' '+w_ans[0]
with open('../hard_neg_dfcd_top20.pkl','w') as f:
  pickle.dump(q_neg,f)



