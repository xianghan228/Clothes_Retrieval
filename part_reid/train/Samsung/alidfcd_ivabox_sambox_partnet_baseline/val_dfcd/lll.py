#!/usr/bin/env python

f=open('evaLib.txt','r')
f1=open('evaLib_1.txt','a')
f2=open('evaLib_2.txt','a')
f3=open('evaLib_3.txt','a')
f4=open('evaLib_4.txt','a')

lines=f.readlines()
for i in range(len(lines)):
  if i<len(lines)/4:
    f1.write(lines[i])
  elif i<len(lines)/2:
    f2.write(lines[i])
  elif i<len(lines)*3/4:
    f3.write(lines[i])
  else:
    f4.write(lines[i])



