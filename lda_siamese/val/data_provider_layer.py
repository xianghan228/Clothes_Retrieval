import numpy as np
import caffe
from PIL import Image
import random
import json

class DataProvider:
  def __init__(self, batchsize=96, img_size=227, path='/data1/qtang/samsung/part_train/ali298.txt'):
    self.batchsize = batchsize
    self.img_size  = img_size
    self.path      = path
    self.img_dict  = self.read_imgs()
    self.batch_len = None
  
  def read_imgs(self):
    img_dict = {}
    f = open(self.path,'r')
    for line in f.readlines():
      img_dict[line.strip().split()[0]]={}
      img_dict[line.strip().split()[0]]['label']=int(line.strip().split()[1])
      img_dict[line.strip().split()[0]]['img_name']=line.strip().split()[0].split('/')[-1]
    return img_dict
  '''
  def read_imgs(self,p1=56,p2=84,p3=140,p4=168):
    img_dict = {}
    f = open(self.path,'r')
    for line in f.readlines():
      img = Image.open(line.strip().split()[0])
      img_name = line.strip().split()[0].split('/')[-1]
      label    = int(line.strip().split()[1])
      img = img.resize((self.img_size,self.img_size))
      img_np = np.asarray(img)
      img_split=[]
      img_split.append(Image.fromarray(img_np[:p2,:,:]))
      img_split.append(Image.fromarray(img_np[p1:p4,:,:]))
      img_split.append(Image.fromarray(img_np[p3:,:,:]))
      img_dict[img_name]={}
      img_dict[img_name]['label'] = label
      for i in range(3):
        img_i    = img_split[i].resize((self.img_size,self.img_size))
        img_np_i = np.asarray(img_i).astype(np.float32)
        img_np_i = img_np_i[:,:,::-1]
        img_np_i = img_np_i.transpose(2,0,1)
        img_np_i = img_np_i-np.load('../mean_227.npy')[0]
        img_dict[img_name]['img_np_'+str(i+1)] = img_np_i
 
    return img_dict
  '''
  def get_batch_vec(self):
    if self.batch_len is None:
      self.batch_len = len(self.img_dict)
      self.batch_index = 0
      self.epoch_counter = 0
      self.img_list = self.img_dict.keys()
      random.shuffle(self.img_list)
    img_np = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    labels  = np.zeros(self.batchsize).astype(np.float32)
    img_name = []
    counter = 0

    while counter<self.batchsize:
      img = Image.open(self.img_list[self.batch_index])
      img = img.resize((self.img_size,self.img_size))
      img_np_indx = np.asarray(img).astype(np.float32)
      img_np_indx = img_np_indx[:,:,::-1]
      img_np_indx = img_np_indx.transpose(2,0,1)
      img_np_indx = img_np_indx-np.load('../mean_227.npy')[0]
      label  = self.img_dict[self.img_list[self.batch_index]]['label']
      img_np[counter,:,:,:] = img_np_indx
      labels[counter]       = label
      img_name.append(self.img_dict[self.img_list[self.batch_index]]['img_name'][:-4])
      if self.batch_index < self.batch_len-1:
        self.batch_index += 1
      else:
        self.epoch_counter += 1
        self.batch_index = 0
        random.shuffle(self.img_list)
      counter += 1
    return img_np, labels, self.epoch_counter, img_name

class DataProviderLayer(caffe.Layer):
  '''
  Deploy part operation after conv2
  '''
  def setup(self,bottom,top):
    param = json.loads(self.param_str)
    print param
    self.batchsize = param['batchsize']
    self.newsize  = param['newsize']
    
    top[0].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[1].reshape(self.batchsize)
    
    self.dp = DataProvider(batchsize=self.batchsize,img_size=self.newsize)

  def reshape(self,bottom,top):
    pass

  def forward(self,bottom,top):
    img_np, labels, _ , _= self.dp.get_batch_vec()
    top[0].data[...] = img_np
    top[1].data[...] = labels

  def backward(self,top,propagate_down,bottom):
    pass

