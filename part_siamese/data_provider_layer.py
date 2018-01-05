import numpy as np
import caffe
from PIL import Image
import random
import json
import os
import pdb
class DataProvider:
  def __init__(self, batchsize=96, img_size=227, path='/data1/qtang/samsung/part_train/ali298.txt'):
    self.batchsize = batchsize
    self.img_size  = img_size
    self.path      = path
    self.img_dict   = self.read_imgs()
    self.batch_len = None
    print 'initialization done...'
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
  def read_imgs(self):
    label_dict = {}
    img_dict = {}
    train_dict = {}
    f = open(self.path,'r')
    for i in range(298):
      label_dict[i] = []
    for line in f.readlines():
      label_dict[int(line.strip().split()[1])].append(line.strip().split()[0])
      img_dict[line.strip().split()[0]] = int(line.strip().split()[1])
    print 'label_dict ok; img_dict ok'
    for img in img_dict:
      label = img_dict[img]
      dif_label = []
      dif_label += label_dict.keys()
      dif_label.remove(label)
      dif_ = random.sample(dif_label,2)
      train_dict[img+';'+random.sample(label_dict[dif_[0]],1)[0]] = str(label)+';'+str(dif_[0])
      train_dict[img+';'+random.sample(label_dict[dif_[1]],1)[0]] = str(label)+';'+str(dif_[1])
      print 'writing dif_pair..'
      same_label = []
      same_label += label_dict[label]
      same_label.remove(img)
      same_ = random.sample(same_label, 2)
      train_dict[img+';'+same_[0]] = str(label)+';'+str(label)
      train_dict[img+';'+same_[1]] = str(label)+';'+str(label)
    print 'writing same_pair..'
    pdb.set_trace()
    return train_dict

  def get_batch_vec(self,p1=56,p2=84,p3=140,p4=168):
    if self.batch_len is None:
      self.batch_len = len(self.img_dict)
      self.batch_index = 0
      self.epoch_counter = 0
      self.img_list = self.img_dict.keys()
      random.shuffle(self.img_list)

    img_nps_1   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    img_p_nps_1   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    img_nps_2   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    img_p_nps_2   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    img_nps_3   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    img_p_nps_3   = np.zeros((self.batchsize,3,self.img_size,self.img_size)).astype(np.float32)
    labels  = np.zeros(self.batchsize).astype(np.float32)
    labels_p  = np.zeros(self.batchsize).astype(np.float32)
    img_names = []
    img_p_names = []
    counter = 0

    while counter<self.batchsize:
      img = Image.open(self.img_list[self.batch_index].split(';')[0])
      img_p = Image.open(self.img_list[self.batch_index].split(';')[1])
      img_name = os.path.basename(self.img_list[self.batch_index].split(';')[0])[:-4]
      img_p_name = os.path.basename(self.img_list[self.batch_index].split(';')[1])[:-4]
      img   = img.resize((self.img_size,self.img_size))
      img_p = img_p.resize((self.img_size,self.img_size))
      img_np = np.asarray(img)
      img_p_np = np.asarray(img_p)
      img_split=[]
      img_split.append(Image.fromarray(img_np[:p2,:,:]))
      img_split.append(Image.fromarray(img_np[p1:p4,:,:]))
      img_split.append(Image.fromarray(img_np[p3:,:,:]))
      img_p_split=[]
      img_p_split.append(Image.fromarray(img_p_np[:p2,:,:]))
      img_p_split.append(Image.fromarray(img_p_np[p1:p4,:,:]))
      img_p_split.append(Image.fromarray(img_p_np[p3:,:,:]))
      img_np_i=[0,0,0]
      img_p_np_i = [0,0,0]
      for i in range(3):
        img_i    = img_split[i].resize((self.img_size,self.img_size))
        img_np_i[i] = np.asarray(img_i).astype(np.float32)
        img_np_i[i] = img_np_i[i][:,:,::-1]
        img_np_i[i] = img_np_i[i].transpose(2,0,1)
        img_np_i[i] = img_np_i[i]-np.load('../mean_227.npy')[0]
        img_p_i    = img_p_split[i].resize((self.img_size,self.img_size))
        img_p_np_i[i] = np.asarray(img_p_i).astype(np.float32)
        img_p_np_i[i] = img_p_np_i[i][:,:,::-1]
        img_p_np_i[i] = img_p_np_i[i].transpose(2,0,1)
        img_p_np_i[i] = img_p_np_i[i]-np.load('../mean_227.npy')[0]
      img_nps_1[counter,:,:,:]   = img_np_i[0]
      img_p_nps_1[counter,:,:,:]   = img_p_np_i[0]
      img_nps_2[counter,:,:,:]   = img_np_i[1]
      img_p_nps_2[counter,:,:,:]   = img_p_np_i[1]
      img_nps_3[counter,:,:,:]   = img_np_i[2]
      img_p_nps_3[counter,:,:,:]   = img_p_np_i[2]
      labels[counter]          = self.img_dict[self.img_list[self.batch_index]].split(';')[0]
      labels_p[counter]          = self.img_dict[self.img_list[self.batch_index]].split(';')[1]
      img_names.append(img_name)
      img_p_names.append(img_p_name)
      if self.batch_index < self.batch_len-1:
        self.batch_index += 1
      else:
        self.epoch_counter += 1
        self.img_dict = read_imgs()
        self.batch_index = 0
        random.shuffle(self.img_list)
      counter += 1
    return img_nps_1, img_nps_2, img_nps_3, labels, img_p_nps_1,img_p_nps_2,img_p_nps_3, labels_p,self.epoch_counter, img_names

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
    top[1].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[2].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[3].reshape(self.batchsize)
    top[4].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[5].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[6].reshape(self.batchsize,3,self.newsize,self.newsize)
    top[7].reshape(self.batchsize)
    top[8].reshape(self.batchsize)
    self.dp = DataProvider(batchsize=self.batchsize,img_size=self.newsize)
  def reshape(self,bottom,top):
    pass

  def forward(self,bottom,top):
    img_nps_1, img_nps_2, img_nps_3, labels, img_p_nps_1, img_p_nps_2,img_p_nps_3,labels_p,_ ,_= self.dp.get_batch_vec()
    top[0].data[...] = img_nps_1
    top[1].data[...] = img_nps_2
    top[2].data[...] = img_nps_3
    top[3].data[...] = labels
    top[4].data[...] = img_p_nps_1
    top[5].data[...] = img_p_nps_2
    top[6].data[...] = img_p_nps_3
    top[7].data[...] = labels_p
    top[8].data[...] = (labels==labels_p).astype(np.float32)
  def backward(self,top,propagate_down,bottom):
    pass

