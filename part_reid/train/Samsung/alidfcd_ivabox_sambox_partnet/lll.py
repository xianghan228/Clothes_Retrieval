#!/usr/bin/env python

f = open('./train_list.txt','r')
f1 = open('./train_list_reverse.txt','a')

lines = f.readlines()

for line in lines[::-1]:
  f1.write(line)
