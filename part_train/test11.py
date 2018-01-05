#!/usr/bin/python
# -*- coding:utf-8 -*-
#from npz to pkl
import argparse
import os as os
import shutil as shutil
import pandas as pd
import numpy as np
import re
import time
import pdb
import pickle
# runnnn('92_v1_96')
# runnnn('36_v1_119')
# runnnn('62_v1_114')
# runnnn('61_v1_64')
# runnnn('15_v1_46')
# runnnn('80_v1_39')
# runnnn('73_v1_9')
# runnnn('95_v1_136')
# runnnn('68_v1_119')
#'feat_eval_siamese_npz_{}_lre-3_train_vggs_part{}/'.format(args.save_path_name,

class test(object):
    
    def __init__(self, loss, step, method):
        self.loss = loss
        self.step = step
        self.method = method
        parser=argparse.ArgumentParser(description='nicainicainicai')
        parser.add_argument('--save_path_name',default=None,help='such as 68_v1_10000')
        parser.add_argument('--save_txt_name',default='result_73.txt',help='such as result.txt') 
        args=parser.parse_args()
        path_name=str(self.loss)
        base_path='/data3/chenliangyu/'
        query_txt_path=base_path+'intermediate/'+'valid_image.txt'
        self.query_feat_path=base_path+'intermediate/'+'query_npz'+path_name+'/'
        self.feat_path = base_path+'intermediate/inference'+path_name + '/' #########################################
        result_path = base_path+'result/distance_siamese_pkl_siamese_'+ path_name +'/'##############################################
        pkl_path = base_path+'result/result_siamese_siamese_'+ path_name +'__pkl/'###################################################
        false_path = base_path+'result/result' + path_name + '__siamese/'
        ImgC_path = ['ImgC_eval_227_part1','ImgC_eval_227_part2','ImgC_eval_227_part3','ImgC_eval_227_part4','img_query_1417']
        '''
        base_path='/home/hsong/shenzhenwei/alibaba/'
        path_name='pool6'
        query_txt_path=base_path+'intermediate/valid_image.txt'
        query_feat_path=base_path+'intermediate/feat_query_vgg_'+path_name+'_part4/' ###############################
        feat_path=base_path+'intermediate/feat_eval_vgg_'+path_name+'/'#########################################
        result_path=base_path+'result/distance_vgg_pkl__'+path_name+'/'##############################################
        pkl_path=base_path+'result/result_vgg_'+path_name+'__pkl/'###################################################
        false_path=base_path+'result/temp_vgg_'+path_name+'__vgg/'
        ImgC_path=['ImgC_eval_227_part1','ImgC_eval_227_part2','ImgC_eval_227_part3','ImgC_eval_227_part4']
        '''
        if not os.path.exists(false_path):
            os.makedirs(false_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        cls_num=[1]#chose a cls to cal
        cls_list=['20015','20033','20036','20061','20062','20068','20073','20080','20092','20095']
        cls_num=[cls_list.index('20033')]



        self.BT=time.time()
        querytxt=open(query_txt_path,'r')
        allquerys=querytxt.readlines()
        query_img_list=[]
        for line in allquerys:
            line=line.strip()
            line=re.split('\W',line)
            line=[name_i for name_i in line if name_i !='']
            query_img_list.append(line[0])
        print('we have {} imgs to query'.format(len(query_img_list)))
        query_part_dict={}
        for num in cls_num:
            query_part_dict[cls_list[num]]=[img for img in query_img_list if img[0:5]==str(int(cls_list[num])+100)]
            print(len(query_part_dict[cls_list[num]]))
            print('...........query_part_dict..............')
        print('we will query {} parts'.format(len(query_part_dict)))
        print(query_part_dict)
        for key in query_part_dict:
            self.retrieved_img_list=query_part_dict[key]
            print(len(self.retrieved_img_list))

            retrieved_img_info_dict=self.get_query_img_feat_mat(self.retrieved_img_list)
            distance_pkl_path= result_path+key+'.pkl'
            #pdb.set_trace()
            self.calculate_distance_multi_img(retrieved_img_info_dict=retrieved_img_info_dict,distance_pkl_path=distance_pkl_path)
        #2
        #from pkl to pkl
        #pkl dict.keys=['distance', 'sortedIndex', 'retrieved_img', 'eval_img', 'discribe']
        #(37,70,20),(37,70,20),(70,),(37,),()
        # we will get 1417 pkl for every pic
        querytxt=open(query_txt_path,'r')
        allquerys=querytxt.readlines()
        querytxt.close()
        false=0
        false_csv=pd.DataFrame(index=range(1417),columns=range(21))
        imgc_time=time.time()
        imgc_clock=time.clock()
        ImgC_list=[]
        for ImgC in ImgC_path:
            img_child_path_list=os.listdir(base_path+'data/'+ImgC)
            for img_child_path in img_child_path_list:
                imgs=os.listdir(base_path+'data/'+ImgC+'/'+img_child_path)
                for img  in imgs:
                    ImgC_list.append(base_path+'data/'+ImgC+'/'+img_child_path+'/'+img)
        print(time.time()-imgc_time)
        print(time.clock()-imgc_clock)
        gt_dict={}
        for line in allquerys:
            line=line.strip()
            line=re.split('\W',line)
            line=[name_i for name_i in line if name_i !='']
            gt_dict[line[0]]=line[1:]
        pkl_list=os.listdir(result_path)
        print(pkl_list)

        for pkl_idx,pkl in enumerate( pkl_list):
            print(pkl)

            print('we are prograssing {} / {}'.format(pkl_idx,len(pkl_list)))
            result_dict={}
            distance_pkl=pickle.load(open(os.path.join(result_path,pkl),'rb'))
            len_npz=len(distance_pkl['eval_img'])
            distance_mat=distance_pkl['distance'][0]
            #print(distance_mat.shape)

            sortedIndex_mat=distance_pkl['sortedIndex'][0]
            eval_img_mat=distance_pkl['eval_img'][0]
            for i in range(len_npz-1):
                print(distance_mat.shape)
                print(distance_pkl['distance'][i+1].T.shape)
                distance_mat=np.concatenate((distance_mat,distance_pkl['distance'][i+1]),axis=1)
            sorted_all_idx_mat=np.argsort(distance_mat)[:,:20]
            sorted_distance_mat=np.sort(distance_mat)[:,:20]
            for img in range(len(sorted_all_idx_mat)):
                result_dict['retrieved_img']=distance_pkl['retrieved_img'][img]
                result_dict['distance']=sorted_distance_mat[img]
                result_dict['gt']=gt_dict[result_dict['retrieved_img']]
                ans_img_list=[]
                for idx in sorted_all_idx_mat[img]:
                    q=int(idx)/20
                    r=int(idx)%20
                    q = int(q)
                    eval_idx=distance_pkl['sortedIndex'][q][img,r]
                    eval_img=distance_pkl['eval_img'][q][eval_idx]
                    ans_img_list.append(str(eval_img[-16:-4]))
                result_dict['ans_img']=ans_img_list
                in_or_not=[]
                for ans in result_dict['ans_img']:
                    if ans in result_dict['gt']:
                        ans_idx=list(result_dict['gt']).index(ans)+1
                        in_or_not.append(ans_idx)
                    else:
                        in_or_not.append(0)
                if sum(in_or_not[0:20])==0:
                    false_csv.iloc[false,0]=result_dict['retrieved_img']
                    false_csv.iloc[false,1:]=result_dict['ans_img']
                    find_time=time.time()
                    find_clock=time.clock()
                    #oldpath=[im for im in ImgC_list if im[-16:-4] in result_dict['ans_img']]
                    #print time.time()-find_time
                    #print time.clock()-find_clock
                    #newpath=false_path+result_dict['retrieved_img']
                    #if not os.path.exists(newpath):
                    #    os.mkdir(newpath)
                    #for ans_img in oldpath:
                    #    shutil.copyfile(ans_img,newpath+'/'+ans_img[-16:])
                    false=false+1
                result_dict['in_or_not']=in_or_not
                #print in_or_not
                pkl_file=open(pkl_path+result_dict['retrieved_img']+'.pkl','wb')
                pickle.dump(result_dict,pkl_file,-1)
                pkl_file.close()
        print('no match ans :',false)
        false_csv.to_csv(false_path+'false_alex.csv',index=false)
        #3
        result_path=base_path+'result/queryScore/'
        if not os.path.exists(result_path):
          os.makedirs(result_path)




        result_txt_path_=base_path+'data/'+args.save_txt_name
        result_txt_=open(result_txt_path_,'a')
        result_txt_1 = open('/data3/chenliangyu/data/ali298.txt','a')
        #result_txt_.write(args.save_path_name)
        #result_txt_.write('.'+str(self.loss))
        result_txt_.write('..'+str(self.loss)+'..'+str(self.step)+'.'+self.method+'...')
        result_txt_1.write('..'+str(self.loss)+'..'+str(self.step)+'.'+self.method+'...')
        for _ in range (21):
            pkl_list=os.listdir(pkl_path)
            len_pkl=len(pkl_list)
            topK_csv=pd.DataFrame(index=range(len_pkl),columns=range(21))
            topD_csv=pd.DataFrame(index=range(len_pkl),columns=range(43))
            topD_num_csv=pd.DataFrame(index=range(len_pkl),columns=range(43))
            AP_csv=pd.DataFrame(index=range(len_pkl),columns=range(2))
            AP_csv_top=pd.DataFrame(index=range(len_pkl),columns=range(2))
            for pkl_idx,pkl in enumerate( pkl_list):
                self.path=os.path.join(pkl_path,pkl)
                name=pkl[0:-4]
                topK_csv.iloc[pkl_idx,0]=name
                topD_csv.iloc[pkl_idx,0]=name
                AP_csv.iloc[pkl_idx,0]=name
                AP_csv_top.iloc[pkl_idx,0]=name
                topK_np,topD_num_np,topD_np=self.cal_p()
                topK_csv.iloc[pkl_idx,1:]=topK_np
                topD_csv.iloc[pkl_idx,1:]=topD_np
                topD_num_csv.iloc[pkl_idx,1:]=topD_num_np
                AP_csv.iloc[pkl_idx,1]=self.cal_ap()
                AP_csv_top.iloc[pkl_idx,1]=self.cal_top(n=_)
            topK_csv.to_csv(result_path+'topK_vgg_'+path_name+'_.csv',index=False)
            topD_csv.to_csv(result_path+'topD_vgge_'+path_name+'_.csv',index=False)
            topD_num_csv.to_csv(result_path+'topD_num_vgge_'+path_name+'_.csv',index=False)
            AP_csv.to_csv(result_path+'AP_vgg_'+path_name+'_uery.csv',index=False)
            AP_csv_top.to_csv(result_path+'AP_vgg_'+path_name+'_uery_topn.csv',index=False)
            
            result_path=base_path+'result/queryScore/'
            AP_csv=pd.read_csv(result_path+'AP_vgg_'+ path_name+'_uery.csv')
            cls_id_list=list(AP_csv.iloc[:,0])
            cls_id_list=[str(cls_id)[3:5] for cls_id in cls_id_list]
            AP_csv.iloc[:,0]=cls_id_list
            grouped_ap=AP_csv.groupby('0').mean()

            print(grouped_ap.head(10))
            print('-----------------------')
            print((AP_csv.iloc[:,1]).sum()/1417)
            result_a=(AP_csv.iloc[:,1]).sum()/1417
            
            grouped_ap.to_csv(result_path+'AP_vgg_'+path_name+'_10cls.csv',index=False)




            ap_csv=pd.read_csv(result_path+'AP_vgg_'+ path_name+'_uery.csv')
            AP_csv_top=pd.read_csv(result_path+'AP_vgg_'+ path_name+'_uery_topn.csv')
            picked_img=pickle.load(open('/data3/chenliangyu/data/'+'pickled_img.pkl','rb'))
            
            c_id='33'
            print(ap_csv.head())
            p_img_list=[]
            p_img_idx_list=[]
            img_list=list(ap_csv.iloc[:,0])
            img_list_idx_img=[img_ for img_ in img_list if str(img_)[0:5]==('201'+c_id)]
            img_list_idx_list=[img_list.index(int(img_in)) for img_in in img_list_idx_img]
            print(len(img_list_idx_list))
            key = '33'
            p_img_list.extend(picked_img[key])
            print(key)
            print(len(p_img_list))
            for p_img in p_img_list:
                idx=img_list.index(int(p_img))
                p_img_idx_list.append(idx)
            value_p=(ap_csv.iloc[p_img_idx_list,1]).mean()
            not_p_list=[i for i in img_list_idx_list if i not in p_img_idx_list]
            print(len(not_p_list))
            
            value_not_p=(ap_csv.iloc[not_p_list,1]).mean()
            if _ == 4:

                result_txt_.write(str(value_not_p)+',')
                result_txt_1.write(str(ap_csv.iloc[img_list_idx_list,1].mean())+',')
            print(value_p)
            print('----------------')
            print(value_not_p)
            print('-------------------')
            print(ap_csv.iloc[img_list_idx_list,1].mean())
            value_all=ap_csv.iloc[img_list_idx_list,1].mean()
            
            
            #result_txt_path_=base_path+'intermediate/'+args.save_txt_name
            #result_txt_=open(result_txt_path_,'a')
            #result_txt_.write(args.save_path_name)
            #result_txt_.write(',')
            #result_txt_.write('train,'+str(value_p)+',')
            #result_txt_.write('test,'+str(value_not_p)+',')
            #result_txt_.write('all,'+str(value_all)+',')
            #result_txt_.close()
            
            c_id='33'
            print(AP_csv_top.head())
            p_img_list=[]
            p_img_idx_list=[]
            img_list=list(AP_csv_top.iloc[:,0])
            img_list_idx_img=[img_ for img_ in img_list if str(img_)[0:5]==('201'+c_id)]
            img_list_idx_list=[img_list.index(int(img_in)) for img_in in img_list_idx_img]
            print(len(img_list_idx_list))
            key = '33'
            #print(picked_img[key])

            p_img_list.extend(picked_img[key])
            print(key)
            print(len(p_img_list))
            for p_img in p_img_list:
                idx=img_list.index(int(p_img))
                p_img_idx_list.append(idx)
            value_p=(AP_csv_top.iloc[p_img_idx_list,1]).mean()
            not_p_list=[i for i in img_list_idx_list if i not in p_img_idx_list]
            print(len(not_p_list))
            
            value_not_p=(AP_csv_top.iloc[not_p_list,1]).mean()
            print(value_p)
            print('----------------')
            print(value_not_p)
            print('-------------------')
            print(AP_csv_top.iloc[img_list_idx_list,1].mean())
            value_all=AP_csv_top.iloc[img_list_idx_list,1].mean()
            
            
            result_txt_.write(str(value_not_p)+',')
            result_txt_1.write(str(AP_csv_top.iloc[img_list_idx_list,1].mean())+',')
        result_txt_.write('\n')
        result_txt_1.write('\n')
        result_txt_.close()
        result_txt_1.close()




    def parse_args():
      parser=argparse.ArgumentParser(description='nicainicainicai')
      parser.add_argument('--save_path_name',default=None,help='such as 68_v1_10000')
      parser.add_argument('--save_txt_name',default='result_73.txt',help='such as result.txt') 
      args=parser.parse_args()
      return args


    #csvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # def query_img_feat(retrieved_img,query_feat_path=query_feat_path):
    #     query_c=retrieved_img[0:5]
    #     query_feat_csvs=os.listdir(query_feat_path)
    #     for query_feat in query_feat_csvs:
    #         if query_feat[-9:-4]==query_c:
    #             imgs_feat_csv=pd.read_csv(query_feat_path+query_feat)
    #             img_names=list(imgs_feat_csv.iloc[:,0])
    #             img_names=[name[-16:-4] for name in img_names]
    #             query_idx=img_names.index(retrieved_img)
    #             retrieved_img_feat_i=np.array(imgs_feat_csv.iloc[query_idx,1:])
    #     return retrieved_img_feat_i
    # def get_query_img_feat_mat(retrieved_img_list,query_feat_path=query_feat_path):
    #     query_idx_list=[]
    #     query_img_c_list=[]
    #     retrieved_img_info_dict={}
    #     query_c=retrieved_img_list[0][0:5]
    #     query_feat_csvs=os.listdir(query_feat_path)
    #     for query_feat in query_feat_csvs:
    #         if query_feat[-9:-4]==query_c:
    #             imgs_feat_csv=pd.read_csv(query_feat_path+query_feat)
    #             img_names=list(imgs_feat_csv.iloc[:,0])
    #             img_names=[name[-16:-4] for name in img_names]
    #             for retrieved_img in retrieved_img_list:
    #                 query_idx=img_names.index(retrieved_img)
    #                 query_idx_list.append(query_idx)
    #                 query_img_c_list.append(retrieved_img)
    #             retrieved_img_feat_mat=np.array(imgs_feat_csv.iloc[query_idx_list,1:])
    #             retrieved_img_info_dict['img_name_list']=query_img_c_list
    #             #retrieved_img_info_dict['img_idx_list']=query_idx_list
    #             retrieved_img_info_dict['img_feat_mat']=retrieved_img_feat_mat
    #     return retrieved_img_info_dict
    #csvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    def get_query_img_feat_mat(self, retrieved_img_list):
        #pdb.set_trace()
        query_idx_list=[]
        query_img_c_list=[]
        retrieved_img_info_dict={}
        query_c=retrieved_img_list[0][0:5]
        query_feat_npzs=os.listdir(self.query_feat_path)
        for query_feat in query_feat_npzs:
            cls_=re.findall('\d{5}',query_feat)
            #cls_='20033'
            if str(int(cls_[0])+100)==query_c:
                imgs_feat_npz=np.load(self.query_feat_path+query_feat)
                img_names=imgs_feat_npz['img']
                img_names=[name[-16:-4] for name in img_names]
                print(len(img_names))

                for retrieved_img in retrieved_img_list:
                    query_idx=img_names.index(retrieved_img)
                    query_idx_list.append(query_idx)
                    query_img_c_list.append(retrieved_img)
                retrieved_img_feat_mat=imgs_feat_npz['feat'].T[query_idx_list,:]
                retrieved_img_info_dict['img_name_list']=query_img_c_list
                retrieved_img_info_dict['img_feat_mat']=retrieved_img_feat_mat
        return retrieved_img_info_dict
    def calculate_distance_multi_img(self, retrieved_img_info_dict,distance_pkl_path):
        cls_name=str(int(retrieved_img_info_dict['img_name_list'][0][0:5])-100)
        npz_path=os.path.join(self.feat_path,cls_name)
        df_len=len(os.listdir(npz_path))*20
        distance_dict={}
        distance_dict['sortedIndex']=[]
        distance_dict['eval_img']=[]
        distance_dict['distance']=[]
        distance_dict['retrieved_img']=retrieved_img_info_dict['img_name_list']
        for npz_i_idx,npz_i in enumerate(os.listdir(npz_path)):
            print('we r searching npz_path:',npz_path,'     ',npz_i_idx+1,'/',df_len/20)
            npz=np.load(os.path.join(npz_path,npz_i))
            print(os.path.join(npz_path,npz_i))
            print(npz_i)
            print(npz_i_idx)
            print(npz)
            print(npz['feat'])
            eval_feats=npz['feat']
            retrieved_img_feat=retrieved_img_info_dict['img_feat_mat']
            print(str(retrieved_img_feat) + 'retrieved_img_feat')
            print(np.isnan(retrieved_img_feat).sum())
            print(retrieved_img_feat.shape)
            print(eval_feats.shape)
            dist=1-np.dot(retrieved_img_feat,eval_feats)/(np.dot((((retrieved_img_feat**2).sum(axis=1))**0.5).reshape(-1,1),(((eval_feats.T**2).sum(axis=1))**0.5).reshape(1,-1)))
            #ones_r=np.ones(np.shape(retrieved_img_feat))
            #ones_e=np.ones(np.shape(eval_feats))
            #dist=np.dot(retrieved_img_feat**2,ones_e.T)+np.dot(ones_r,(eval_feats**2).T)-2*np.dot(retrieved_img_feat,(eval_feats.T))
            print('we ve finish calculating this npz ,waiting {} mins'.format((time.time()-self.BT)/60))
            sortedIndex_mat=np.argsort(dist)#np.shape(sortedIndex)=n,8192
            sorted_dist_mat=np.sort(dist)
            nan_num=np.isnan(dist).sum()
            if nan_num!=0:
                find_nan=((eval_feats**2).sum(axis=1))
                print('num eval_feats 0 dot:', (find_nan==0).sum())
                print('num retrieved 0 dot:', (((retrieved_img_feat**2).sum(axis=1))==0).sum())
                print(np.shape(find_nan))
                print('npz',npz_i)
                print('a!!'*100)
            print('we ve finish sort the mat ,waiting {} mins'.format((time.time()-self.BT)/60))
            print(sortedIndex_mat.shape)
            distance_dict['distance'].append(sorted_dist_mat[:,:20])
            distance_dict['sortedIndex'].append(sortedIndex_mat[:,:20])
            distance_dict['eval_img'].append(npz['img'])
        print('we ve made a ',len(distance_dict['sortedIndex']),'pkl',distance_pkl_path)
        distance_dict['discribe']='sortedIndex is correspondence to eval_img ; '
        pkl_file=open(distance_pkl_path,'wb')
        pickle.dump(distance_dict,pkl_file,-1)
        pkl_file.close()

    # creat query img_list




    def cal_p(self):
        #pkl_dict=pickle.load(open(path,'rb'), encoding='iso-8859-1')
        pkl_dict=pickle.load(open(self.path,'rb'))
        in_or_not=pkl_dict['in_or_not']
        distance=pkl_dict['distance']
        topK_np=(np.zeros(20)).astype(int)
        topD_np=(np.zeros(42)).astype(int)
        topD_num_np=(np.zeros(42))
        for idx,i in enumerate( in_or_not):
            D=int(100*distance[idx])
            if D<10:
                idx_d=0
            elif D>50:
                idx_d=41
            else:
                idx_d=D-9
            topD_num_np[idx_d]+=1
            if i>0:
                topK_np[idx]=topK_np[idx]+1
                D=int(100*distance[idx])
                if D<10:
                    idx_d=0
                elif D>50:
                    idx_d=41
                else:
                    idx_d=D-9
                topD_np[idx_d]+=1
        return topK_np,topD_num_np,topD_np


    def cal_ap(self):
        #pkl_dict=pickle.load(open(path,'rb'), encoding='iso-8859-1')
        pkl_dict=pickle.load(open(self.path,'rb'))
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


    def cal_top (self,n=4):
        p=0.
        pkl_dict=pickle.load(open(self.path,'rb'))
        in_or_not = pkl_dict['in_or_not']
        if sum(in_or_not[:n])>0:
            p=1.
        return p


        # pkl_list=os.listdir(pkl_path)
        # len_pkl=len(pkl_list)
        # topK_csv=pd.DataFrame(index=range(len_pkl),columns=range(21))
        # topD_csv=pd.DataFrame(index=range(len_pkl),columns=range(43))
        # topD_num_csv=pd.DataFrame(index=range(len_pkl),columns=range(43))
        # AP_csv=pd.DataFrame(index=range(len_pkl),columns=range(2))
        # for pkl_idx,pkl in enumerate( pkl_list):
        #     path=os.path.join(pkl_path,pkl)
        #     name=pkl[0:-4]
        #     topK_csv.iloc[pkl_idx,0]=name
        #     topD_csv.iloc[pkl_idx,0]=name
        #     AP_csv.iloc[pkl_idx,0]=name
        #     topK_np,topD_num_np,topD_np=cal_p(path)
        #     topK_csv.iloc[pkl_idx,1:]=topK_np
        #     topD_csv.iloc[pkl_idx,1:]=topD_np
        #     topD_num_csv.iloc[pkl_idx,1:]=topD_num_np
        #     AP_csv.iloc[pkl_idx,1]=cal_p(path)
        # topK_csv.to_csv(result_path+'topK_alex.csv',index=False)
        # topD_csv.to_csv(result_path+'topD_alex.csv',index=False)
        # topD_num_csv.to_csv(result_path+'topD_num_alex.csv',index=False)
        # AP_csv.to_csv(  result_path+'AP_alex.csv',index=False)
        #
        #
        #
        #
        #