#!/usr/bin/env python
f = open('./train_list_org.txt','r')
lines = f.readlines()
f1 = open('./train_list_org_f.txt','a')
f2 = open('./train_list_plus.txt','a')
for line in lines:
  if '/20133' in '/'+line.split()[0].split('/')[-1]:
    f2.write(line)
  else:
    f1.write(line)
