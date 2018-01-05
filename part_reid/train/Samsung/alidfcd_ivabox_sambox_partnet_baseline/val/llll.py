#!/usr/bin/env python

f = open('/data1/chenliangyu/intermediate/valid_image_update333662.txt','r')
f33 = open('./query_gt_33_new.txt','a')
f36 = open('./query_gt_36_new.txt','a')
f62 = open('./query_gt_62_new.txt','a')
lines = f.readlines()

for line in lines:
  if line.startswith('20133'):
    f33.write(line)
  elif line.startswith('20136'):
    f36.write(line)
  elif line.startswith('20162'):
    f62.write(line)
