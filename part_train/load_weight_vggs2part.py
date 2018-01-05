#!/usr/bin/env python
import caffe
import numpy as np
if __name__ == '__main__':
  caffe.set_mode_gpu()
  vggs=caffe.Net('/data1/qtang/samsung/part_train/vggs.prototxt','/data1/zwshen/alibaba/train_MM_2/pool6_siamese_train/models.caffemodel',caffe.TEST)
  part=caffe.Net('/data1/qtang/samsung/part_train/part_train_withfc7_.prototxt',caffe.TEST)

  for j in range(5):
      for i in range(3):
        part.params['conv'+str(j+1)+'_'+str(i+1)][0] = vggs.params['conv'+str(j+1)][0]
        part.params['conv'+str(j+1)+'_'+str(i+1)][1] = vggs.params['conv'+str(j+1)][1]
  for i in range(3):
    part.params['fc6'+'_'+str(i+1)][0] = vggs.params['fc6'][0]
    part.params['fc7'+'_'+str(i+1)][0] = vggs.params['fc7'][0]
  part.save('/data1/qtang/samsung/part_train/part_train_withfc7_.caffemodel')

