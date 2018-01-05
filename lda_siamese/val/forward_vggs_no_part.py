#!/usr/bin/env python
import caffe
from data_provider_layer import DataProvider
import os
import pdb
import numpy as np
import mlab
from mlab.releases import latest_release as matlab
matlab.path(matlab.path(),'/data1/qtang/samsung/wheels/matlab')

def feat_extract(device_id=2, img_list_path='', batchsize=256, net='./deploy_lda_siamese.prototxt', model='./197_iter_500.caffemodel', iter=500 , blob='pool6'):
  f_path = './'+'feat_'+os.path.basename(img_list_path).split('.')[0]+'/iter'+str(iter)+'/'+blob
  if not os.path.exists(f_path):
    os.makedirs(f_path)
  dp = DataProvider(batchsize=batchsize, path=img_list_path)
  img_list_len=len(dp.img_dict)
  net = caffe.Net(net, model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  if blob == 'fc7_n':
    feature = np.zeros((batchsize*100,298))
  else:
    feature = np.zeros((batchsize*100,8192))
  img_name = []
  block_label = []
  index = 0
  epoch = 0
  counter = 1
  while epoch==0:
    img_np, labels, epoch, img_name_batch = dp.get_batch_vec()
    net.blobs['top_0'].data[...] = img_np
    net.forward()
    index += 1
    feature[batchsize*(index-1):batchsize*index,:] = net.blobs[blob].data.reshape(batchsize,-1)
    img_name+=img_name_batch
    block_label+=list(labels)
    if epoch==1:
      block_feat  = feature[:(img_list_len-(counter-1)*batchsize*100),:]
      block_feat /= np.sqrt((block_feat**2).sum(axis=1)).reshape(-1,1)
      block_label = block_label[:(img_list_len-(counter-1)*batchsize*100)]
      img_name = img_name[:(img_list_len-(counter-1)*batchsize*100)]
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path, feat=block_feat, img=img_name)
    
    elif index == 100:
      block_feat  = feature
      block_feat /= np.sqrt((block_feat**2).sum(axis=1)).reshape(-1,1)
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path,feat=block_feat,img=img_name)
      index = 0
      img_name = []
      block_label = []
      counter +=1
    
if __name__ == '__main__':
  feat_extract(img_list_path='/data1/qtang/samsung/train_no_part/evaLib.txt')
  #feat_extract(img_list_path='/data1/qtang/samsung/part_train/query33ali.txt')
  #feat_extract(img_list_path='/data1/qtang/samsung/part_train/ali298.txt')
  

