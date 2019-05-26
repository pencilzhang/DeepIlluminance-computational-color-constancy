#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:59:48 2017

@author: tzheng
"""
import scipy.io as sio
import numpy as np 
work_path='/data/list/'

def create_lmdb(function,function_txt):
    function=function
    train=open(function_txt,'r')
    namelist=open(work_path+'list/namelist.txt','r')
    labellist=open(work_path+'list/labellist.txt','r')
    namelist=namelist.readlines()
    labellist=labellist.readlines()
    
    #label=np.uint8(np.zeros(3*len(labellist)).reshape(len(labellist),3))
    label=np.uint8(np.zeros(3)).reshape(1,3)
    for line in train:
        imagename=line[:-5]
        for i in range(len(namelist)):   
            if namelist[i][0:8]==imagename:
                f=open(function+'.txt','a')
                f.write(namelist[i])
                f.close()            
                #ll=labellist[i][-1]
                ll=labellist[i].split()
                label=np.row_stack((label,np.uint8(ll)))
    label=np.delete(label,0,axis=0)
#    label1= np.matrix(label)
    sio.savemat(function+'.mat', {'label':label}) 

       

create_lmdb('train','image190.txt')
create_lmdb('val','image189_1.txt')
create_lmdb('test','image189_2.txt')

