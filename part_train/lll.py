#!/usr/bin/env python

f=open('./ali298.txt','r')

fn=open('./query33ali_.txt','a')

for line in f.readlines():
  if line.strip().split()[0].split('/')[-1].startswith('20133'):
    fn.write(line)

