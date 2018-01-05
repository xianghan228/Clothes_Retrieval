#!/usr/bin/env python
f=open('./train_list.txt','r')
fn = open('../val/ali197.txt','a')
197_list=[]
for line in f.readlines():
  if 'query' in line:
    path=line.split()[0]
    197_list+=os.listdir(path)


