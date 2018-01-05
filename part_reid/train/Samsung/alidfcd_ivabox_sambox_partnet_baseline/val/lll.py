#!/usr/bin/env python
fn  =  open('../train_list_plus.txt','a')
subs = ['36','62']
train_data = {}
q_gts = {}
with open('../train_list.txt','r') as f:
  for line in f.readlines():
    train_data[line.split()[0].split('/')[-1][:-4]] = line.split()[1]

for sub in subs:
  f1 = open('./query_gt_'+sub+'.txt','r')
  f2 = open('./query'+sub+'.txt','r')
  lines1 = f1.readlines()
  for line1 in lines1:
    q_gts[line1.split(',')[0]] = line1.split(',')[1].split(';')

  lines2 = f2.readlines()
  for line2 in lines2:
    query = line2.split()[0]
    q_img = query.split('/')[-1][:-4]
    for gt in q_gts[q_img]:
      if gt in train_data:
        fn.write(query+' '+train_data[gt]+'\n')
        break
