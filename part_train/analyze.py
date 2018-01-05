#!/usr/bin/env python
import caffe
import numpy as np
from data_provider_layer import DataProvider

net=caffe.Net('./part_val.prototxt', '/data1/qtang/samsung/part_train/models/new_fc6weights/_iter_8000.caffemodel', caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(2)

dp      = DataProvider(batchsize=64)
img_len = len(dp.img_dict)
feat    = np.zeros((img_len,298))

epoch         = 0
batch_counter = 0
while epoch == 0:
  img_np_1,img_np_2,img_np_3,labels, epoch, img_name = dp.get_batch_vec()
  net.blobs['top_0'].data[...] = img_np_1
  net.blobs['top_1'].data[...] = img_np_2
  net.blobs['top_2'].data[...] = img_np_3
  net.blobs['top_3'].data[...] = labels
  net.forward()
  batch_counter += 1
  if epoch == 1:
    feat[64*(batch_counter-1):,:] = net.blobs['fc7'].data[:img_len-64*batch_counter,:]
  else:
    feat[64*(batch_counter-1):64*batch_counter,:] = net.blobs['fc7'].data

feat /= np.sqrt((feat**2).sum(axis=1)).reshape(-1,1)
var   = np.var(feat, axis=1)
mean  = np.mean(feat, axis=1)
max_  = np.max(feat,axis=1)
min_  = np.min(feat,axis=1)
np.save('./var_part',var)
np.save('./mean_part',mean)
np.save('./max_part',max_)
np.save('./min_part',min_)




