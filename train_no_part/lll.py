#!/usr/bin/env python
f = open('ImgC_eval_227.txt','r')
fn = open('evaLib.txt','w')
for line in f.readlines():
  fn.write('/data1/chenliangyu/data/ImC/'+'ImgC_eval_227'+line.strip().split('ImgC_eval_227')[1]+'\n')


