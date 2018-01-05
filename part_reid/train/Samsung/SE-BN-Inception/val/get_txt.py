#!/usr/bin/env python
import os

path_query = '/data1/zwshen/samsung/intermediate/forward_img/ali_sambox/img_query_1417/200360'
path_evalib = '/data1/zwshen/samsung/intermediate/forward_img/ali_sambox/ImgC_eval_227_part1'
path_evaLib = '/data1/zwshen/samsung/intermediate/forward_img/ali_sambox_big/ImgC_eval_227_part1'

f1=open('./query36_new.txt','a')
f2=open('./evalib36_new.txt','a')
f3=open('./evaLib36_new.txt','a')

for file_name in os.listdir(path_query):
  f1.write(os.path.join(path_query,file_name)+' 0\n')


for folder in os.listdir(path_evalib):
  if '20036' in folder:
    path = os.path.join(path_evalib,folder)
    for file_name in os.listdir(path):
      f2.write(os.path.join(path,file_name)+' 0\n')

for folder in os.listdir(path_evaLib):
  if '20036' in folder:
    path = os.path.join(path_evaLib,folder)
    for file_name in os.listdir(path):
      f3.write(os.path.join(path,file_name)+' 0\n')
