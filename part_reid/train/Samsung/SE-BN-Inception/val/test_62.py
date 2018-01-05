#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os 
import shutil 
import pandas as pd
import numpy as np
import re
import time
import pdb
import pickle



def  get_query_img_and_gtdict(query_txt_path):

    querytxt=open(query_txt_path,'r')
    allquerys=querytxt.readlines()
    query_img_list=[]
    gt_dict={}
    for line in allquerys:
        line=line.strip()
        p1=line.split(',')[0]
        p2=line.split(',')[1].split(';')
        
        p2=[name_i for name_i in p2 if name_i !='']
        query_img_list.append(p1)
        gt_dict[p1]=p2
    print 'we have {} imgs to query'.format(len(query_img_list))
    return query_img_list,gt_dict


def get_query_img_feat_mat(retrieved_img_list,query_feat_path_list):
    query_idx_list=[]
    query_img_c_list=[]
    retrieved_img_info_dict={}
    for query_feat in query_feat_path_list:
        imgs_feat_npz=np.load(query_feat)
        img_names = []
        for img_name in imgs_feat_npz['img']:
          img_names+=[img_name.split()[0].split('/')[-1][:-4]]
        for retrieved_img in retrieved_img_list:
            query_idx=img_names.index(retrieved_img)
            query_idx_list.append(query_idx)
            query_img_c_list.append(retrieved_img)
        retrieved_img_feat_mat=imgs_feat_npz['feat'][query_idx_list,:]
        retrieved_img_info_dict['img_name_list']=query_img_c_list
        retrieved_img_info_dict['img_feat_mat']=retrieved_img_feat_mat
    return retrieved_img_info_dict
    
def calculate_distance_multi_img(retrieved_img_info_dict,distance_pkl_path,feat_npz_path_list):

    distance_dict={}
    distance_dict['sortedIndex']=[]
    distance_dict['eval_img']=[]
    distance_dict['distance']=[]
    distance_dict['retrieved_img']=retrieved_img_info_dict['img_name_list']

    for npz_i_idx,npz_i in enumerate(feat_npz_path_list):
        print 'we r searching npzs:',npz_i,'     ',npz_i_idx+1,'/',len(feat_npz_path_list)
        npz=np.load(npz_i)
        eval_feats=npz['feat']
        retrieved_img_feat=retrieved_img_info_dict['img_feat_mat']
        print np.isnan(retrieved_img_feat).sum()
        eval_feats         /= np.sqrt((eval_feats*eval_feats).sum(axis=1)).reshape(-1,1)
        retrieved_img_feat /= np.sqrt((retrieved_img_feat**2).sum(axis=1)).reshape(-1,1)
        dist=1-np.dot(retrieved_img_feat,eval_feats.T)/(np.dot((((retrieved_img_feat**2).sum(axis=1))**0.5).reshape(-1,1),(((eval_feats**2).sum(axis=1))**0.5).reshape(1,-1)))
        print 'we ve finish calculating this npz '
        sortedIndex_mat=np.argsort(dist)
        sorted_dist_mat=np.sort(dist)
        nan_num=np.isnan(dist).sum()
        if nan_num!=0:
            find_nan=((eval_feats**2).sum(axis=1))
            print 'num eval_feats 0 dot:', (find_nan==0).sum()
            print 'num retrieved 0 dot:', (((retrieved_img_feat**2).sum(axis=1))==0).sum()
            print np.shape(find_nan)
            print 'npz',npz_i
            print 'a!!'*100
        print sortedIndex_mat.shape
        distance_dict['distance'].append(sorted_dist_mat[:,:40])
        distance_dict['sortedIndex'].append(sortedIndex_mat[:,:40])
        img_names=[]
        for img_name in npz['img']:
          img_names+=[img_name.split()[0].split('/')[-1][:-4]]
        distance_dict['eval_img'].append(img_names)
    print 'we ve made a ',len(distance_dict['sortedIndex']),'pkl',distance_pkl_path
    distance_dict['discribe']='sortedIndex is correspondence to eval_img ; '
    pkl_file=open(distance_pkl_path,'wb')
    pickle.dump(distance_dict,pkl_file,-1)
    pkl_file.close()

def rank20imgs(distance_pkl_path,gt_dict,pkl_path):
    result_dict={}
    distance_pkl=pickle.load(open(distance_pkl_path,'rb'))
    len_npz=len(distance_pkl['eval_img'])
    distance_mat=distance_pkl['distance'][0]
    sortedIndex_mat=distance_pkl['sortedIndex'][0]
    eval_img_mat=distance_pkl['eval_img'][0]

    for i in range(len_npz-1):
        distance_mat=np.concatenate((distance_mat,distance_pkl['distance'][i+1]),axis=1)
    sorted_all_idx_mat=np.argsort(distance_mat)[:,:40]
    sorted_distance_mat=np.sort(distance_mat)[:,:40]
    for img in range(len(sorted_all_idx_mat)):
        result_dict['retrieved_img']=distance_pkl['retrieved_img'][img]
        result_dict['distance']=sorted_distance_mat[img]
        result_dict['gt']=gt_dict[result_dict['retrieved_img']]
        ans_img_list=[]
        for idx in sorted_all_idx_mat[img]:
            q=int(idx)/40
            r=int(idx)%40
            eval_idx=distance_pkl['sortedIndex'][q][img,r]
            eval_img=distance_pkl['eval_img'][q][eval_idx]
            ans_img_list.append(os.path.basename(eval_img))
        result_dict['ans_img']=ans_img_list
        in_or_not=[]
        for ans in result_dict['ans_img']:
            if ans in result_dict['gt']:
                ans_idx=list(result_dict['gt']).index(ans)+1
                in_or_not.append(ans_idx)
            else:
                in_or_not.append(0)
        result_dict['in_or_not']=in_or_not
        pkl_file=open(os.path.join(pkl_path,result_dict['retrieved_img']+'.pkl'),'wb')
        pickle.dump(result_dict,pkl_file,-1)
        pkl_file.close()
  
def cal_ap(path):

    pkl_dict=pickle.load(open(path))
    in_or_not=pkl_dict['in_or_not']
    denominator=len(pkl_dict['gt'])
    if denominator>20:
        denominator=20
    numerator=0
    nn=0
    for idx,i in enumerate(in_or_not):
        if i >0:
            nn+=1
            numerator+=(nn/float(idx+1))
    ap=numerator/denominator
    return ap

def cal_top (path,n=4):
    p=0.
    pkl_dict=pickle.load(open(path,'r'))
    in_or_not = pkl_dict['in_or_not']
    if sum(in_or_not[:n])>0:
        p=1.
    return p


def img_ap(pkl_path,save_csv_path,name_flag=''):
    pkl_list=os.listdir(pkl_path)
    pkl_list=[_ for _ in pkl_list if '.pkl' in _ ]
    len_pkl=len(pkl_list)
    AP_csv=pd.DataFrame(index=range(len_pkl),columns=range(2))
    AP_csv_top=pd.DataFrame(index=range(len_pkl),columns=range(2))
    for pkl_idx,pkl in enumerate( pkl_list):
        path=os.path.join(pkl_path,pkl)
        name=pkl[0:-4]
        AP_csv.iloc[pkl_idx,0]=name
        AP_csv_top.iloc[pkl_idx,0]=name
        AP_csv.iloc[pkl_idx,1]=cal_ap(path)
        AP_csv_top.iloc[pkl_idx,1]=cal_top(path,n=4)
    AP_csv.to_csv(os.path.join(save_csv_path,name_flag+'_AP_alimap20.csv'),index=False)
    AP_csv_top.to_csv(os.path.join(save_csv_path,name_flag+'_AP_topk.csv'),index=False)
    
def write2txt(save_csv_path,name_flag='',pickled_img_pkl_path=None,save_result_txt=None):
    AP_csv=pd.read_csv(os.path.join(save_csv_path,name_flag+'_AP_alimap20.csv'))
    AP_csv_top=pd.read_csv(os.path.join(save_csv_path,name_flag+'_AP_topk.csv'))
    pickled_img=pickle.load(open(pickled_img_pkl_path,'rb'))
    p_img_list=pickled_img['p_img']
    notp_img_list=pickled_img['not_p_img']
    img_list=list(AP_csv.iloc[:,0])
    img_list=[str(_) for _ in img_list]
    
    p_img_idx_list=[]
    notp_img_idx_list=[]


    for p_img in p_img_list:
        idx=img_list.index(str(p_img))
        p_img_idx_list.append(idx)
    value_p_ali_ap=(AP_csv.iloc[p_img_idx_list,1]).mean()
    value_p_top_ap=(AP_csv_top.iloc[p_img_idx_list,1]).mean()

    for notp_img in notp_img_list:
        idx=img_list.index(str(notp_img))
        notp_img_idx_list.append(idx)
    value_not_p_ali_ap=(AP_csv.iloc[notp_img_idx_list,1]).mean()
    value_not_p_top_ap=(AP_csv_top.iloc[notp_img_idx_list,1]).mean()
    
    value_ali_ap_all=value_p_ali_ap*len(p_img_idx_list)+value_not_p_ali_ap*len(notp_img_idx_list)
    value_top_ap_all=value_p_top_ap*len(p_img_idx_list)+value_not_p_top_ap*len(notp_img_idx_list)
    value_ali_ap_all/=(len(p_img_idx_list)+len(notp_img_idx_list))
    value_top_ap_all/=(len(p_img_idx_list)+len(notp_img_idx_list))

    txt=open(save_result_txt,'a')
    txt.write(name_flag)
    txt.write('|')
    txt.write(str(value_top_ap_all))
    txt.write('\n')
    txt.close()
    os.system('cat {}'.format(save_result_txt))




if __name__ =='__main__':
    def parse_args():
      parser=argparse.ArgumentParser(description='nicainicainicai')
      parser.add_argument('--base_path_name',     default='./result/ap',help='')
      parser.add_argument('--pkl_path',           default='./result/pkl/62',help='')
      parser.add_argument('--query_txt_path',     default='./query_gt_62_new.txt',help='')
      parser.add_argument('--pickled_img_pkl_path',default='/data1/zwshen/samsung/intermediate/62_pickled_img.pkl',help='')
      parser.add_argument('--query_feat_path',    default='./feat_query62_baseline/iter80000/normed_feature',help='')
      parser.add_argument('--eval_feat_path',     default='./feat_evaLib62_baseline/iter80000/normed_feature',help='')
      parser.add_argument('--name_flag',          default='iter80000_Lib_62_baseline_update',help='such as 68_v1_10000')
      parser.add_argument('--distance_pkl_path',   default='./result/dis/62/distance_pkl_',help='')
      parser.add_argument('--save_result_txt',     default='result_baseline.txt',help='such as result.txt') 
      args=parser.parse_args()
      return args
  
    args=parse_args()
    
    query_feat_path_list=os.listdir(args.query_feat_path)
    feat_npz_path_list=os.listdir(args.eval_feat_path)

    query_feat_path_list=[os.path.join(args.query_feat_path,_) for _ in query_feat_path_list]
    feat_npz_path_list=[os.path.join(args.eval_feat_path,_ ) for _ in feat_npz_path_list]
    

    name_flag=args.name_flag
    save_result_txt=args.save_result_txt

    pickled_img_pkl_path=args.pickled_img_pkl_path

    distance_pkl_path=args.distance_pkl_path+args.name_flag+'.pkl'
    pkl_path=os.path.join(args.pkl_path,args.name_flag)
    if not os.path.exists(pkl_path):
      os.makedirs(pkl_path)
    save_csv_path=pkl_path
    save_result_txt=os.path.join(args.base_path_name,args.save_result_txt)
    query_txt_path=args.query_txt_path  
    
    retrieved_img_list,gt_dict=get_query_img_and_gtdict(query_txt_path)  
    retrieved_img_info_dict=get_query_img_feat_mat(retrieved_img_list,query_feat_path_list)
    calculate_distance_multi_img(retrieved_img_info_dict,distance_pkl_path,feat_npz_path_list)
    rank20imgs(distance_pkl_path,gt_dict,pkl_path)
    img_ap(pkl_path,save_csv_path,name_flag)
    write2txt(save_csv_path,name_flag,pickled_img_pkl_path,save_result_txt=save_result_txt)



