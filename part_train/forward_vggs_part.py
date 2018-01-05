#!/usr/bin/env python
import caffe
from data_provider_layer import DataProvider
import os
import numpy as np
def feat_extract(device_id=2, img_list_path='', batchsize=64, net='./part_val.prototxt', model='./models/vggs_fc6weights/_iter_7600.caffemodel'):
  f_path = '/data1/qtang/samsung/part_train/'+'feat_'+os.path.basename(img_list_path).split('.')[0]+'/vggs_fc6weights/iter7600/concate'
  if not os.path.exists(f_path):
    os.makedirs(f_path)
  dp = DataProvider(batchsize=batchsize, path=img_list_path)
  net = caffe.Net(net, model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  feature = np.zeros((batchsize*100,12288))
  img_name = []
  index = 0
  epoch = 0
  counter = 1
  while epoch==0:
    img_nps_1, img_nps_2, img_nps_3, labels, epoch, img_name_batch= dp.get_batch_vec()
    net.blobs['top_0'].data[...] = img_nps_1
    net.blobs['top_1'].data[...] = img_nps_2
    net.blobs['top_2'].data[...] = img_nps_3
    net.blobs['top_3'].data[...] = labels
    net.forward()
    index += 1
    feature[batchsize*(index-1):batchsize*index,:] = net.blobs['concate'].data
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
  feat_extract(img_list_path='/data1/qtang/samsung/part_train/query33ali.txt')




