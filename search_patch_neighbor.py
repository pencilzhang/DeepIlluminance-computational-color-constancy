#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:13:47 2017

@author: tzheng
"""

from PIL import Image
import scipy.io as sio  
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
#import math
import random

work_path = '/data/'
image_path = work_path + 'image_8bit/'
mask_path = work_path + 'mask/'
white_black_pixel = work_path + 'bright_dark_pixel/'
white_black_pixel_patch = work_path + 'patch/'
white_black_pixel_patch_0_5neighbor=work_path + 'neighbor/'

f = open(work_path + 'filelist.txt')
names=f.readlines()
rgbs=np.uint8(sio.loadmat(work_path+'list/remake_label.mat')['label'])

patch_size=224
neighbor_size=patch_size/2
new=time.time()

def Neighborhood(h,w,neighbor_size,big_image,path):
    h1=512+h-neighbor_size
    w1=512+w-neighbor_size           

    img_patch_neighbor=big_image[h1:(h1+patch_size + 2*neighbor_size),
                                 w1:(w1+patch_size + 2*neighbor_size),:]

    im_neighbor=Image.fromarray(img_patch_neighbor)
    im_neighbor.save(path + name+'_'+str(w)+'_'+str(h)+'.png')
    return 0

k=1
percent=0.035
num_1,complete_num=214,1
for num in range(len(names)):
    start=time.time()
    name=names[num][:-2]
    rgb=rgbs[num]
    img = np.uint8(plt.imread(image_path + name + '.png')*255)
    mask = np.uint8(plt.imread(mask_path + name + '.png' )*255)#mask is 255
    print ''
    print 'image name : ',name
    #calculated average vector
    ave_r,ave_g,ave_b,useful_pixel_amount,mask_pixel_amount=0,0,0,np.float32(0),0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if mask[j][i]==0:
                ave_r+=img[j][i][0]
                ave_g+=img[j][i][1]
                ave_b+=img[j][i][2]
                useful_pixel_amount+=1
            else:
                mask_pixel_amount+=1
                img[j][i]=0
    print 'useful pixel amount=',useful_pixel_amount
    print 'mask pixel amount=',mask_pixel_amount
    ave_r,ave_g,ave_b = ave_r/useful_pixel_amount, ave_g/useful_pixel_amount,ave_b/useful_pixel_amount
    mean_pixel=np.array([ave_r,ave_g,ave_b])
    print 'mean pixel=',mean_pixel
    time1=time.time()
    print time1-start
    
    #calculated projection
    maxmin_pixel_amount=np.uint32(np.round(useful_pixel_amount*percent))
    distance=np.float32(np.zeros(img.shape[1]*img.shape[0]).reshape(img.shape[0],
                        img.shape[1])-1)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if mask[j][i]==0:
                distance[j][i]=(img[j][i].dot(mean_pixel))/(np.sqrt(mean_pixel.dot(mean_pixel)))                     
    time2=time.time()
    print time2-time1
    
    distance_order=distance.reshape(-1)
    order=distance_order.argsort()
    luminance_max=distance_order[order[-1*maxmin_pixel_amount]]
    luminance_min=distance_order[order[maxmin_pixel_amount+mask_pixel_amount]]
    
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if distance[j][i] >=luminance_max:
                mask[j][i]=240
            elif (distance[j][i] <= luminance_min) and (distance[j][i] > -1):            
                mask[j][i]=50

    im=Image.fromarray(mask)
    im.save(white_black_pixel + name + '.png')
    time3=time.time()
    print time3-time2    
      
    exist=0    
    big_image=np.uint8(np.zeros((img.shape[0]+1024)*(img.shape[1]+1024)*3).reshape((img.shape[0]+1024),
                                (img.shape[1]+1024),3))
    big_image[512:(512+img.shape[0]),512:(512+img.shape[1]),:]=img
    for i in range(1000):
        w=random.randint(0,img.shape[1]-patch_size)
        h=random.randint(0,img.shape[0]-patch_size)            
        patch=mask[h:h + patch_size,w:w+patch_size]
        patch_1d=patch.reshape(-1) 
        high=0
        low=0
        if 255 in patch_1d:
            continue
        if 50 in patch_1d:
            low=1
        if 240 in patch_1d:
            high=1
        if low ==1 and high ==1:
            exist+=1
            img_patch=img[h:h+patch_size,w:w+patch_size,:]
            im=Image.fromarray(img_patch)
            im.save(white_black_pixel_patch + name+'_'+str(w)+'_'+str(h)+'.png')
            img_patch_neighbor=Neighborhood(h,w,neighbor_size,big_image,white_black_pixel_patch_0_5neighbor)
            
            f=open(work_path + 'list/list'+'.txt','a')
            f.write(name+'_'+str(w)+'_'+str(h)+'.png'+' '+str(rgb[0])+' '+str(rgb[1])+' '+str(rgb[2])+'\n')
            f.close()
            
            n=open(work_path + 'list/namelist'+'.txt','a')
            n.write(name+'_'+str(w)+'_'+str(h)+'.png'+' '+str(0)+'\n')
            n.close()
            
            g=open(work_path + 'list/labellist'+'.txt','a')
            g.write(str(rgb[0])+' '+str(rgb[1])+' '+str(rgb[2])+'\n')
            g.close()
            print exist
        if exist == complete_num:
            break
    
    f=open(work_path + 'list/number'+'.txt','a')
    f.write(name+'='+str(exist)+'\n')
    f.close()
    #hat does't meet the requirements in problem/, so the bright-dark percent should be relaxed
    if exist == 0:
        im.save(work_path +'problem/' + name + '.png')
    time4=time.time()
    print time4-time3
old=time.time()
print 'total time : ',old-new         