import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import numpy as np


'''
def get_label_dict(path_txt='/data1/zwshen/samsung/intermediate/valid_image.txt'):
  lines=open(path_txt).readlines()
  dict_={}
  i=0
  for lin in lines:
    lin=lin.strip()
    if '20133' == lin[:5]:
      dict_[lin.split(',')[0]]=i
      for im in (lin.split(',')[1]).split(';'):
        dict_[im]= i
    i+=1
  return dict_
'''
def get_trian_array_and_label(path_npz='/home/qtang/storage/samsung/lda_siamese/val/feat_ali298/iter500/pool6/1.npz'):
  npz=np.load(path_npz)
  feat=npz['feat']
  img=npz['img']
  label=npz['label']
  label_array=np.array(label)
  print feat.shape
  print label_array.shape
  return feat,label_array

def save_array(base_path,feat,im):
  np.savez(base_path,feat=feat,img=im)
  



lda = LinearDiscriminantAnalysis(n_components=298) 
x,y=get_trian_array_and_label()
lda.fit(x,y)
#x2=lda.transform(x)
print lda.score(x,y)
'''
from_path='/home/qtang/storage/samsung/lda_siamese/val/feat_query33ali/iter500/pool6'
to_path='/home/qtang/storage/samsung/lda_siamese/val/feat_query33ali/iter500/pool6_lda'
if not os.path.exists(to_path):
  os.makedirs(to_path)
for npz in os.listdir(os.path.join(from_path)):
  tp=np.load(os.path.join(from_path,npz))
  feat=tp['feat']
  print 'trans feat:',feat.shape
  feat2=lda.transform(feat)
  print 'to feat2:',feat2.shape
  img=tp['img']
  save_array(os.path.join(to_path,npz),feat2,img)
'''

from_path='/home/qtang/storage/samsung/lda_siamese/val/feat_evaLib/iter500/pool6'
to_path='/home/qtang/storage/samsung/lda_siamese/val/feat_evaLib/iter500/pool6_lda'
if not os.path.exists(to_path):
  os.makedirs(to_path)
for npz in os.listdir(os.path.join(from_path)):
  tp=np.load(os.path.join(from_path,npz))
  feat=tp['feat']
  print 'trans feat:',feat.shape
  feat2=lda.transform(feat)
  print 'to feat2:',feat2.shape
  img=tp['img']
  save_array(os.path.join(to_path,npz),feat2,img)
  
  
