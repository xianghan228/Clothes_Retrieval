#!/usr/bin/env python
from PIL import Image,ImageFont,ImageDraw
import os
import pickle 
import pdb
font = ImageFont.truetype("/usr/share/fonts/smc/Meera.ttf", 15)
pkl_path = '/data1/qtang/samsung/part_reid/train/Samsung/SE-BN-Inception/val/result/pkl/62/iter75000_Lib_62_sambox'
with open('../query62.txt','r') as f:
  querys = f.readlines()
with open('../evaLib62_new.txt','r') as f:
  evas   = f.readlines()

for pkl_file in os.listdir(pkl_path):
  if '.pkl' in pkl_file:
    pkl = pickle.load(open(os.path.join(pkl_path,pkl_file),'r'))
    query = pkl['retrieved_img']
    gts   = pkl['gt'][:4]
    ans   = pkl['ans_img'][:4]
    in_or_not    = pkl['in_or_not'][:4]
    
    for query_path in querys:
      if query in query_path:
        query = query_path.split()[0]
    
    for i in range(4):
      for eva_path in evas:
        if gts[i] in eva_path:
          gts[i] = eva_path.split()[0]
        if ans[i] in eva_path:
          ans[i] = eva_path.split()[0]
  
    toImage   = Image.new('RGB',(224*5,224*3),(255,255,255))
    fromImage = Image.open(query).resize((224,224))
    loc       = (0,0)
    toImage.paste(fromImage,loc)
    
    flag = 0
    for i in range(4):
      fromImage = Image.open(gts[i]).resize((224,224))
      loc       = (224*i,224)
      toImage.paste(fromImage,loc)

      fromImage = Image.open(ans[i]).resize((224,224))
      draw      = ImageDraw.Draw(fromImage)
      if in_or_not[i]==0:
        text = 'wrong'
      else:
        text = 'right'
        flag = 1
      draw.text((210,210),text,font=font,fill='red')
      loc       = (224*i,224*2)
      toImage.paste(fromImage,loc)
      
    draw = ImageDraw.Draw(toImage)
    text1 = 'Query'
    text2 = 'Ground Truth'
    text3 = 'Top4'
    draw.text((224,0),text1,font=font,fill=(0,0,0,0))
    draw.text((224*4,224),text2,font=font,fill=(0,0,0,0))
    draw.text((224*4,224*2),text3,font=font,fill=(0,0,0,0))
    if flag == 0:
      toImage.save('/data1/qtang/samsung/part_reid/train/Samsung/SE-BN-Inception/val/visualization/img_wrong/'+query.split('/')[-1])
    
