#!/usr/bin/env python
import caffe
from data_provider_layer import DataProvider
import os
import numpy as np
def feat_extract(device_id=0, img_list_path='', batchsize=150, flag='', net='./deploy.prototxt', model='', iter=100000 , blob='pool5_93'):
  f_path = './'+'feat_'+os.path.basename(img_list_path).split('.')[0]+flag+'/'+blob
  if not os.path.exists(f_path):
    os.makedirs(f_path)
  dp = DataProvider(batchsize=batchsize, path=img_list_path)
  img_list_len=len(dp.img_dict)
  net = caffe.Net(net, model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  feature = np.zeros((batchsize*100,8192))
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
  for i in [75000]:
    feat_extract(img_list_path='../../SE-BN-Inception/val/evaLib62.txt',model='./attr.caffemodel',iter=i,flag='_atrr')
    feat_extract(img_list_path='../../SE-BN-Inception/val/evalib62.txt',model='./attr.caffemodel',iter=i,flag='_attr')
    feat_extract(img_list_path='../../SE-BN-Inception/val/query62.txt',model='./attr.caffemodel',iter=i,flag='_attr')
  
