#!/usr/bin/env python
import numpy as np
import caffe
from data_provider_layer import DataProvider
import time
import ipdb

def test_acc(device_id, model, prefix=''):
  net = caffe.Net('./part_val.prototxt', model, caffe.TEST)
  caffe.set_device(device_id)
  caffe.set_mode_gpu()
  dp = DataProvider(batchsize = 64, img_size=227)

  epoch = 0
  loss_list  = []
  acc_layer  = []
  while epoch == 0:
    img_nps_1, img_nps_2, img_nps_3, labels, epoch = dp.get_batch_vec()
    net.blobs['top_0'].data[...] = img_nps_1
    net.blobs['top_1'].data[...] = img_nps_2
    net.blobs['top_2'].data[...] = img_nps_3
    net.blobs['top_3'].data[...] = labels
    net.forward()
    loss_list.append(net.blobs['loss'].data)
    acc_layer.append(net.blobs['accuracy'].data)
  mean_loss = np.array(loss_list).mean()
  acc       = np.array(acc_layer).mean()

  return acc, mean_loss

if __name__ == '__main__':
  iterid = [7000,7200,7400,7600,7800,8000]
  for it in iterid:
    acc, mean_loss = test_acc(device_id=7, model='./models/_iter_%d.caffemodel'%it, prefix='iteration%d'%it)
    print 'iteration:',it
    print 'mean_loss:',mean_loss
    print 'acc:',acc
    with open('./acc_result.txt','a') as f:
      f.write(str(it)+':  mean_loss: '+str(mean_loss)+' acc: '+str(acc)+'\n')
