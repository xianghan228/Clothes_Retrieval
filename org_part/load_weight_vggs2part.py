#!/usr/bin/env python
import caffe
import numpy as np
if __name__ == '__main__':
  caffe.set_mode_gpu()
  caffe.set_device(2)
  vggs=caffe.Net('/data1/qtang/samsung/train_no_part/deploy_no_part.prototxt','/data1/zwshen/alibaba/train_MM_2/pool6_siamese_train/models.caffemodel',caffe.TEST)
  org_part=caffe.Net('/data1/qtang/samsung/org_part/org_part_train.prototxt',caffe.TEST)

  for j in range(5):
      org_part.params['conv'+str(j+1)][0] = vggs.params['conv'+str(j+1)][0]
      org_part.params['conv'+str(j+1)][1] = vggs.params['conv'+str(j+1)][1]
      for i in range(3):
        org_part.params['conv'+str(j+1)+'_'+str(i+1)][0] = vggs.params['conv'+str(j+1)][0]
        org_part.params['conv'+str(j+1)+'_'+str(i+1)][1] = vggs.params['conv'+str(j+1)][1]
  org_part.params['fc6'][0] = vggs.params['fc6'][0]
  org_part.params['fc6'][1] = vggs.params['fc6'][1]
  for i in range(3):
    org_part.params['fc6'+'_'+str(i+1)][0] = vggs.params['fc6'][0]
    org_part.params['fc6'+'_'+str(i+1)][1] = vggs.params['fc6'][1]
  org_part.save('/data1/qtang/samsung/org_part/org_part.caffemodel')

