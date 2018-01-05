#!/usr/bin/env python
import caffe
from data_provider_layer import DataProvider
import os
import numpy as np
def feat_extract(device_id=2, img_list_path='', batchsize=48, net='./org_part_deploy.prototxt', iter=6000 , blob='fc8'):
  f_path = '/data1/qtang/samsung/org_part/'+'feat_'+os.path.basename(img_list_path).split('.')[0]+'/iter'+str(iter)+'/'+blob
  if not os.path.exists(f_path):
    os.makedirs(f_path)
  dp = DataProvider(batchsize=batchsize, path=img_list_path)
  model='./models/_iter_%d.caffemodel'%iter
  net = caffe.Net(net, model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  if blob == 'concat2':
    feature = np.zeros((batchsize*100,8192))
  else:
    feature = np.zeros((batchsize*100,298))
  img_name = []
  index = 0
  epoch = 0
  counter = 1
  while epoch==0:
    img_np_1,img_np_2,img_np_3,labels,img_np_org,epoch, img_name_batch= dp.get_batch_vec()
    net.blobs['top_0'].data[...] = img_np_1
    net.blobs['top_1'].data[...] = img_np_2
    net.blobs['top_2'].data[...] = img_np_3
    net.blobs['top_3'].data[...] = labels
    net.blobs['top_4'].data[...] = img_np_org
    net.forward()
    index += 1
    feature[batchsize*(index-1):batchsize*index,:] = net.blobs[blob].data
    img_name+=img_name_batch
    if epoch==1:
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path,feat=feature[:index*batchsize,:],img=img_name)
    elif index == 100:
      path = os.path.join(f_path, str(counter)+'.npz')
      np.savez(path,feat=feature,img=img_name)
      index = 0
      img_name = []
      counter +=1
  

if __name__ == '__main__':
  feat_extract(img_list_path='/data1/qtang/samsung/train_no_part/eva33ali298.txt')
  feat_extract(img_list_path='/data1/qtang/samsung/part_train/query33ali.txt')



