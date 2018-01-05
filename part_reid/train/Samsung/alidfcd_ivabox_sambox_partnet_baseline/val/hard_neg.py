#!/usr/bin/env python
import os
import pickle
import numpy as np
import pdb
base_path = '/data1/qtang/samsung/part_reid/train/Samsung/alidfcd_ivabox_sambox_partnet_baseline/val/result/pkl'
sub_paths  = ['33/iter75000_Lib_33_baseline_new','36/iter75000_Lib_36_baseline_new','62/iter80000_Lib_62_baseline_new']
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
  f = open('./evaLib'+sub_path.split('/')[0]+'.txt','r')
  for line in f.readlines():
    if an+'.jpg' in line:
      return line.split()[0]
    



for sub_path in sub_paths:
  path = os.path.join(base_path,sub_path)
  for file_name in os.listdir(path):
    if file_name.endswith('.pkl'):
      pkl_dic = pickle.load(open(os.path.join(path,file_name),'r'))
      query   = pkl_dic['retrieved_img']
      label   = find_q_id(sub_path,query)
      ans     = np.array(pkl_dic['ans_img'])
      idx     = list(np.where(np.array(pkl_dic['in_or_not'])==0)[0])
      if len(idx)!=0:
        w_ans   = list(ans[idx])
       
        w_ans_path = []
        for w_an in w_ans:
          w_ans_path.append(find_an_path(sub_path,w_an))
        q_neg[label] = w_ans_path
        print 'one query done'
        print str(label)+':'+' '+w_ans_path[0]
        print 'len: '+str(len(w_ans_path))
with open('../hard_neg_top40_new_mark.pkl','w') as f:
  pickle.dump(q_neg,f)



