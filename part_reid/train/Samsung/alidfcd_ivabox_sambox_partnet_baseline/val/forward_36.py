#!/usr/bin/env python
import caffe
from data_provider_layer import DataProvider
import os
import numpy as np
import time
def feat_extract(device_id=0, img_list_path='', batchsize=290, flag='', net='./deploy_33.prototxt', model='../snapshot/model_iter_100000.caffemodel', iter=100000 , blob='normed_feature'):
  f_path = './'+'feat_'+os.path.basename(img_list_path).split('.')[0]+flag+'/iter'+str(iter)+'/'+blob
  if not os.path.exists(f_path):
    os.makedirs(f_path)
  dp = DataProvider(batchsize=batchsize, path=img_list_path)
  img_list_len=len(dp.img_dict)
  net = caffe.Net(net, model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  feature = np.zeros((batchsize*100,net.blobs[blob].data.shape[1]))
  img_name = []
  index = 0
  epoch = 0
  counter = 1
  while epoch==0:
    img_np, labels, epoch, img_name_batch= dp.get_batch_vec()
    net.blobs['data'].data[...] = img_np
    net.forward()
    index += 1
    feature[batchsize*(index-1):batchsize*index,:] = net.blobs[blob].data.reshape(batchsize,-1)
    img_name+=img_name_batch
    if epoch==1:
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path,feat=feature[:(img_list_len-(counter-1)*batchsize*100),:],img=img_name[:(img_list_len-(counter-1)*batchsize*100)])
    elif index == 100:
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path,feat=feature,img=img_name)
      index = 0
      img_name = []
      counter +=1

if __name__ == '__main__':
  time.sleep(3600*2)
  for i in [75000,80000,85000,90000]:
    feat_extract(img_list_path='./evaLib36.txt',model='../snapshot_baseline_top40_qf_senet/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf_senet')
    feat_extract(img_list_path='./evalib36.txt',model='../snapshot_baseline_top40_qf_senet/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf_senet')
    feat_extract(img_list_path='./query36.txt',model='../snapshot_baseline_top40_qf_senet/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf_senet')
    feat_extract(img_list_path='./evaLib36.txt',model='../snapshot_baseline_top40_qf/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf')
    feat_extract(img_list_path='./evalib36.txt',model='../snapshot_baseline_top40_qf/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf')
    feat_extract(img_list_path='./query36.txt',model='../snapshot_baseline_top40_qf/model_iter_%d.caffemodel'%i,iter=i,flag='_top40_qf')
