# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:44:14 2017

@author: tzheng
"""
#from pylab import*
caffe_root='/root/caffe-master'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

estimate_rgb='prod'
label='label'
png_PATH='png/'
caffemodel='VGG_ILSVRC_16_layers.caffemodel'
solver=caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(caffemodel)

test_iter = 189    
test_interval = 500    
max_iter = 160000       

train_batch_size= 32
test_batch_size = 15      

import math
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))    
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data); plt.axis('off')
def  Angular_error(prediction,label): 
    a=prediction
    b=label
    c=np.dot(a.T,b)/(np.linalg.norm(a,ord=None) * np.linalg.norm(b,ord=None))
    if c>1:
        c=1
    Angular_error=math.degrees(math.acos(c)) 
    return Angular_error
    
def  Error(prediction,label): 
    error=np.zeros(len(prediction))         
    for i in range(len(prediction)): 
        a=prediction[i].reshape(3,1)
        b=label[i].reshape(3,1)
        error[i]=Angular_error(a,b)
    m_error=np.zeros(len(prediction)+3)
    m_error[0]=np.mean(error[0:len(prediction)])
    d=np.mean(prediction,axis=0)
    e=np.median(prediction,axis=0)
    f=np.mean(label,axis=0)
    m_error[1]=Angular_error(d,f)
    m_error[2]=Angular_error(e,f)   
    m_error[3:(len(prediction)+3)]=error[0:(len(prediction))]      
    return m_error

import time

start=time.time()

train_loss = []
train_average_error = []
train_median_error = []
test_loss = []
test_patch_mean_error = []
test_patch_median_error = []
test_mean_pooling_mean_error = []
test_mean_pooling_median_error = []
test_median_pooling_mean_error = []
test_median_pooling_median_error = []
patch_mean_error = []
patch_median_error = []
mean_pooling_error = []
median_pooling_error = []

per_patch_estimate = np.zeros(test_iter*test_batch_size*3).reshape(test_iter,test_batch_size,3)
per_image_label = np.zeros(test_iter*3).reshape(test_iter,3)

for it in range(max_iter/test_interval):
    solver.step(test_interval)     
    train_loss.append(solver.net.blobs['loss1'].data*1)
    end1=time.time()
    print end1-start        
    a=Error(solver.net.blobs[estimate_rgb].data.reshape(train_batch_size,3),solver.net.blobs[label].data.reshape(train_batch_size,3))
    train_average_error.append(a[0])
    print 'Iteration',it*test_interval,'train_average_error=',train_average_error[-1]
    train_median_error.append(np.median(a[3:(train_batch_size+3)]))
    print 'Iteration',it*test_interval,'train_median_error=',train_median_error[-1]
    loss = 0
    for test_it in range(test_iter):
        solver.test_nets[0].forward()
        Angular=Error(solver.test_nets[0].blobs[estimate_rgb].data, solver.test_nets[0].blobs[label].data.reshape(test_batch_size,3))         
        patch_mean_error.append(Angular[0])
        patch_median_error.append(Angular[3:(test_batch_size+3)])
        mean_pooling_error.append(Angular[1])
        median_pooling_error.append(Angular[2])           
        per_patch_estimate[test_it]=solver.test_nets[0].blobs[estimate_rgb].data
        per_image_label[test_it]=solver.test_nets[0].blobs[label].data[1].reshape(3)            
