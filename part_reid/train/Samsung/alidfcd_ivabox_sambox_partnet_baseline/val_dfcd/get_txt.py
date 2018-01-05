#!/usr/bin/env python
import pdb
f = open('../train_list.txt','r')
lines = f.readlines()
label_img_df = {}
label_img_cd = {}

for line in lines:
  label = line.split()[1]
  file_name = line.split()[0]
  if 'cd' in line:
    if label in label_img_cd:
      label_img_cd[label]+=[file_name]
    else:
      label_img_cd[label] = [file_name]
  elif 'df' in line:
    if label in label_img_df:
      label_img_df[label]+=[file_name]
    else:
      label_img_df[label]=[file_name]

f1 = open('./query_cd.txt','a')
f2 = open('./query_gt_cd.txt','a')
for key in label_img_cd:
  f1.write(label_img_cd[key][0]+' '+key+'\n')
  for file_name in label_img_cd[key]:
    if label_img_cd[key].index(file_name)==0:
      f2.write(file_name+' '+key+',')
    elif label_img_cd[key].index(file_name)==len(label_img_cd[key])-1:
      f2.write(file_name+' '+key+'\n')
    else:
      f2.write(file_name+' '+key+';')

f3 = open('./query_df.txt','a')
f4 = open('./query_gt_df.txt','a')
for key in label_img_df:
  if len(label_img_df[key])>10:
    f3.write(label_img_df[key][0]+' '+key+'\n')
    for file_name in label_img_df[key]:
      if label_img_df[key].index(file_name)==0:
        f4.write(file_name+' '+key+',')
      elif label_img_df[key].index(file_name)==len(label_img_df[key])-1:
        f4.write(file_name+' '+key+'\n')
      else:
        f4.write(file_name+' '+key+';')

f5 = open('./evaLib.txt','a')
f6 = open('../val/evaLib33.txt','r')
f7 = open('../val/evaLib36.txt','r')
f8 = open('../val/evaLib62.txt','r')

lines1 = f6.readlines()
lines2 = f7.readlines()
lines3 = f8.readlines()
lines_ = lines1+lines2+lines3

for line in lines_:
  lines += [line.split()[0]+' '+'46585\n']

print len(lines)


for line in lines:
  f5.write(line)



