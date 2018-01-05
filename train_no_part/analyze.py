#!/usr/bin/env python
import caffe
import numpy as np
from data_provider_layer import DataProvider

net=caffe.Net('./deploy_no_part.prototxt', './models/models_iter_8000.caffemodel', caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(2)

dp      = DataProvider(batchsize=64)
img_len = len(dp.img_dict)
feat    = np.zeros((img_len,298))

epoch         = 0
batch_counter = 0
while epoch == 0:
  img_np, labels, epoch, img_name = dp.get_batch_vec()
  net.blobs['top_0'].data[...] = img_np
  net.blobs['top_1'].data[...] = labels
  net.forward()
  batch_counter += 1
  if epoch == 1:
    feat[64*(batch_counter-1):,:] = net.blobs['fc7_n'].data[:img_len-64*batch_counter,:]
  else:
    feat[64*(batch_counter-1):64*batch_counter,:] = net.blobs['fc7_n'].data

feat /= np.sqrt((feat**2).sum(axis=1)).reshape(-1,1)
var   = np.var(feat, axis=1)
mean  = np.mean(feat, axis=1)
max_  = feat.max(axis=1)
min_  = feat.min(axis=1)
np.save('./var_no_part',var)
np.save('./mean_no_part',mean)
np.save('./max_no_part',max_)
np.save('./min_no_part',min_)



