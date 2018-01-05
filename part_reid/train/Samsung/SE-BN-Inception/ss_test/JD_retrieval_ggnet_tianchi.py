#import cv2
#coding:utf-8
import os
import os.path as osp
import numpy as np
import sys

import caffe
from caffe.proto import caffe_pb2
import lmdb
import argparse
import json
import time
import re
import urllib2,urllib

import pdb
dim = 8192
batch_num=8000
topnum = 4


def Parse_args():
	parser=argparse.ArgumentParser(description="Retrieval")
        parser.add_argument('--rlmdb_path', dest='rLmdbPath',help='retrieval dataset(lmdb)',default='./',type=str)
        parser.add_argument('--qlmdb_path', dest='qLmdbPath',help='test dataset(lmdb)',default='./',type=str)
        parser.add_argument('--rlmdb_path2', dest='rLmdbPath2',help='retrieval dataset(lmdb)',default='./',type=str)
        parser.add_argument('--qlmdb_path2', dest='qLmdbPath2',help='test dataset(lmdb)',default='./',type=str)
        
        parser.add_argument('--fw_result', dest='FW_result',help='result',default='./',type=str)
	args=parser.parse_args()
	return args

def OpenLMDB(Path):
	env_db=lmdb.open(Path)
	txn=env_db.begin()
	cursor=txn.cursor()
	return cursor

def OpenTXT(Path):
	env_db_img=[]
	with open(Path,'r') as f:
		env_db_img=f.readlines()
	f.close()
	return env_db_img

def ParseImgList(FLines,SChar):
	imgList=[]
	for line in FLines:
		img_name, label=line.split(SChar)
		imgList.append(img_name)
	return imgList

def ParseImgLMDB(cursor):
        n = 0
        cc = []
        for index, value in cursor:
            cc.append(index)
#        help(cursor)
#        print cursor.count()
#        help(type(cursor))
#        help (cursor)
#        print type(cursor)
#        print cursor.shape
#        n = len(cursor)
#        n = cursor.count()
        n = len(cc)
        print n
        db_feature = np.zeros((n, dim),dtype=np.float)
        db_label = np.zeros(n)
	for index, value in cursor:
		datum=caffe_pb2.Datum()
		datum.ParseFromString(value)
		feature=caffe.io.datum_to_array(datum)
                label = datum.label
		db_feature[int(index),:]=feature.reshape(dim)
                db_label[int(index)] = label
		if int(index)%100==0:
			print 'current line:%d total line:%d'%(int(index),n)
	return db_feature, db_label

def CalcEuDist(Qdb_feature,Qlabel,Rdb_feature,Rlabel):
    
    EuDist = np.zeros((Qdb_feature.shape[0], Rdb_feature.shape[0]))
    print Qdb_feature.shape[0], Rdb_feature.shape[0]
    for i in range(Qdb_feature.shape[0]):
        SingleCp = np.tile(Qdb_feature[i], [Rdb_feature.shape[0],1])
        d_value = SingleCp - Rdb_feature
        EuDist[i] = np.sum(d_value * d_value, 1)
        if i % 100 == 0:
            print "i = %d" % i
#        for j in range(db_feature.shape[0]):
#            EuDist[i][j] = VecEuDist(sdb_feature[i], db_feature[j])
#            if j % 100 == 0 :
#                print "j = %d" % j
    Top=np.argsort(EuDist,1)
    Result=Top[:,0:topnum]
    TopK20=0
    tnum = 0
    right_tag_file = open('tag.txt','w')
    for PIndex,PerRes in enumerate(Result): # total sample
        qlabel = Qlabel[PIndex]
        rlabel = Rlabel[PerRes]
        if qlabel in rlabel:
            TopK20 += 1
            right_tag_file.write("1\n")
        else:
            right_tag_file.write("0\n")
    print "20 num: %d"%(TopK20)
    return TopK20

def CalcSimDist_xue(Qdb_feature, Rcursor):
    global fw_name
    TmpSndb = np.sqrt(np.sum(Qdb_feature*Qdb_feature, 1)).reshape(Qdb_feature.shape[0], 1)
    Sndb=Qdb_feature/np.tile(TmpSndb, (1, Qdb_feature.shape[1]))
    nn = 0
    cc = []
    for index, value in Rcursor:
      cc.append(index)
    nn = len(cc)
    print 'num of database:'+str(nn)
    
    iter= int(nn/batch_num)
    print "need to iter:"+str(iter)
    Rdb_feature = np.zeros((batch_num, dim),dtype=np.float)
    db_label = np.zeros(batch_num)
    SimDist=-1
    Rlabel=-1
    
    iter_temp=0
    for index_n, value in Rcursor:
      
      index_in=int(index_n)
      index_im=index_in%batch_num
      
      datum=caffe_pb2.Datum()
      datum.ParseFromString(value)
      feature=caffe.io.datum_to_array(datum)
      label = datum.label
      Rdb_feature[index_im,:]=feature.reshape(dim)
      db_label[index_im] = label
      
      if (index_in+1)%batch_num==0:
        iter_temp+=1
        if index_in !=0:
          print 'line num of matrix:%d total line:%d'%(index_in,nn)
          fw=open(fw_name,'a')
          fw.write('line num of matrix:' + str(index_in)+'  total line:'+str(nn)+'\n')
          fw.close()
          if iter_temp==1:
            Rlabel = np.copy(db_label)
          else:
            Rlabel=np.concatenate((Rlabel,db_label),axis=0)
          
          TepNdb=np.sqrt(np.sum(Rdb_feature*Rdb_feature, 1)).reshape(Rdb_feature.shape[0],1)
          Ndb = Rdb_feature/np.tile(TepNdb, (1,Rdb_feature.shape[1]))
          SimDist_temp=np.dot(Sndb, Ndb.transpose())
          
          if iter_temp==1:
            SimDist=np.copy(SimDist_temp)
          else:
            SimDist=np.concatenate((SimDist,SimDist_temp),axis=1)
          #Rdb_feature = np.zeros((batch_num, dim),dtype=np.float)
          #db_label = np.zeros(batch_num)
    overplus=nn%batch_num
    if overplus!=0:
      if iter!=0:
        Rlabel=np.concatenate((Rlabel,db_label[0:overplus]),axis=0)
      else:
        Rlabel=np.copy(db_label[0:overplus])
      Rdb_feature=Rdb_feature[0:overplus]
      
      TepNdb=np.sqrt(np.sum(Rdb_feature*Rdb_feature, 1)).reshape(Rdb_feature.shape[0],1)
      Ndb = Rdb_feature/np.tile(TepNdb, (1,Rdb_feature.shape[1]))
      SimDist_temp=np.dot(Sndb, Ndb.transpose())
      
      if iter!=0:
        SimDist=np.concatenate((SimDist,SimDist_temp),axis=1)
      else:
        SimDist=np.copy(SimDist_temp)
    return Rlabel, SimDist
    
def CalcSimDist(Qdb_feature, Qlabel, Rcursor,Qdb_feature2 , Rcursor2 ):

    Rlabel,SimDist=CalcSimDist_xue(Qdb_feature, Rcursor)
    Rlabel2,SimDist2=CalcSimDist_xue(Qdb_feature2, Rcursor2)
    
    
    SimDist_new=SimDist+SimDist2
    Top = np.argsort(-1*SimDist_new, 1)
    
    
    Result = Top[:,0:topnum]
    TopK20 = 0
    right_tag_file = open('tag.txt','w')
    retrieval_result = open('retrieval_result_down.txt','w')
    result_list = []
    for PIndex,PerRes in enumerate(Result): # total sample
#        retrieval_result.write('%d:'%(PIndex))
        result_list.append(PerRes)
        for index in PerRes:
          retrieval_result.write('%d,'%(index))
        retrieval_result.write('\n')
        qlabel = Qlabel[PIndex]
        rlabel = Rlabel[PerRes]
        if qlabel in rlabel:
            TopK20 += 1
            right_tag_file.write("1\n")
        else:
            right_tag_file.write("0\n")
    retrieval_result.close()
    print "20 num: %d"%(TopK20)
    return TopK20,result_list

def parse_result(result):
    #query_list = open('query.txt')
    #retrieval_list = open('retrieval.txt')
    query_list = open('./crop_list_samsungpay61/query_crop.txt')
    retrieval_list = open('./crop_list_samsungpay61/retrieval_crop.txt')
    query_file = []
    retrieval_file = []
    for record in query_list:
        query_file.append(record[0:-3])
#        print query_file[-1]
    for record in retrieval_list:
        retrieval_file.append(record[0:-3])
    if not os.path.exists('result'):
        os.makedirs('result')
    for i, res in enumerate(result):
        query_name = query_file[i]
        sort_i = 1
        folder = 'result/%d' % i
        if not os.path.exists(folder):
            os.makedirs(folder)

        target_query = 'result/%d/%s' % (i, query_name.split('/')[-1])
        urllib.urlretrieve(qury_name, target_query)
        for index in res:
#        for retrieval_name in retrieval_file[res]:
            retrieval_name = retrieval_file[index]
            target_name =  'result/%d/%d_%s' % (i, sort_i, retrieval_name.split('/')[-1])
            urllib.urlretrieve(retrieval_name, target_name)
            sort_i += 1
            





if __name__=="__main__":


  args=Parse_args()
  rlmdbPath=args.rLmdbPath
  qlmdbPath=args.qLmdbPath
  Qcursor=OpenLMDB(qlmdbPath)
  Rcursor=OpenLMDB(rlmdbPath)
  Qdb_feature,Qlabel=ParseImgLMDB(Qcursor)
  #Rdb_feature,Rlabel=ParseImgLMDB(Rcursor)
  global fw_name
  fw_name=args.FW_result
  
  rlmdbPath2=args.rLmdbPath2
  qlmdbPath2=args.qLmdbPath2
  Qcursor2=OpenLMDB(qlmdbPath2)
  Rcursor2=OpenLMDB(rlmdbPath2)
  Qdb_feature2,Qlabel2=ParseImgLMDB(Qcursor2)
  #Rdb_feature2,Rlabel2=ParseImgLMDB(Rcursor2)
  
  #topK,result_list = CalcSimDist(Qdb_feature,Qlabel,Rdb_feature,Rlabel)
  topK,result_list = CalcSimDist(Qdb_feature,Qlabel,Rcursor,Qdb_feature2,Rcursor2)
  print "Top20: %f" % (float(topK)/Qdb_feature.shape[0])
  fw=open(fw_name,'a')
  fw.write('Top20:' + str(float(topK)/Qdb_feature.shape[0])+'\n')
  fw.close()


