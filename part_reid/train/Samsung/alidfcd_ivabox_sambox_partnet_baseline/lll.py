#!/usr/bin/env python
import pickle

f = open('./train_list_plus.txt','r')

lines = f.readlines()
label_query = {}

for line in lines:
  label_query[int(line.split()[1])] = line.split()[0]

with open('./label_query.pkl','w') as f:
  pickle.dump(label_query,f)

