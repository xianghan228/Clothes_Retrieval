#!/usr/bin/env python
import caffe
import numpy as np
if __name__ == '__main__':
  caffe.set_mode_gpu()
  caffe.set_device(2)
  vggs=caffe.Net('./deploy_no_part.prototxt','/data1/zwshen/alibaba/train_MM_2/pool6_siamese_train/models.caffemodel',caffe.TEST)
  part=caffe.Net('./part_siamese_train.prototxt',caffe.TEST)

  for j in range(5):
      for i in range(3):
        part.params['conv'+str(j+1)+'_'+str(i+1)][0] = vggs.params['conv'+str(j+1)][0]
        part.params['conv'+str(j+1)+'_'+str(i+1)+'_p'][0] = vggs.params['conv'+str(j+1)][0]
        part.params['conv'+str(j+1)+'_'+str(i+1)][1] = vggs.params['conv'+str(j+1)][1]
        part.params['conv'+str(j+1)+'_'+str(i+1)+'_p'][1] = vggs.params['conv'+str(j+1)][1]
  for i in range(3):
    part.params['fc6'+'_'+str(i+1)][0] = vggs.params['fc6'][0]
    part.params['fc6'+'_'+str(i+1)][1] = vggs.params['fc6'][1]
    part.params['fc6'+'_'+str(i+1)+'_p'][0] = vggs.params['fc6'][0]
    part.params['fc6'+'_'+str(i+1)+'_p'][1] = vggs.params['fc6'][1]
  part.save('./part_siamese.caffemodel')

