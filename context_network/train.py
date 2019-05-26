# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:44:14 2017

@author: tzheng
"""
from pylab import*
caffe_root='/root/caffe-master'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize']=(15,15)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

estimate_rgb='fc_8'
label='label'
png_PATH='png/'
caffemodel='context.caffemodel'#this model conv layers is initialized by singlenet(patch and surround)
solver=caffe.SGDSolver('patch_solver.prototxt')
solver.net.copy_from(caffemodel)

test_iter = 189  
test_interval = 2000
max_iter = 160000

train_batch_size= 15
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
train_loss = np.ones(max_iter//test_interval)
train_average_error = np.ones(max_iter//test_interval)
train_median_error = np.ones(max_iter//test_interval)
test_loss=np.ones(max_iter//test_interval)
test_patch_mean_error=np.ones(max_iter//test_interval)
test_patch_median_error=np.ones(max_iter//test_interval)
test_mean_pooling_mean_error=np.ones(max_iter//test_interval)
test_mean_pooling_median_error=np.ones(max_iter//test_interval)
test_median_pooling_mean_error=np.ones(max_iter//test_interval)
test_median_pooling_median_error=np.ones(max_iter//test_interval)
patch_mean_error = np.ones(test_iter)
patch_median_error = np.ones(test_iter*test_batch_size)
mean_pooling_error = np.ones(test_iter)
median_pooling_error = np.ones(test_iter)

per_patch_estimate = np.ones(test_iter*test_batch_size*3).reshape(test_iter,test_batch_size,3)
per_image_label=np.ones(test_iter*3).reshape(test_iter,3)

for it in range(max_iter):
    solver.step(1)  
    
    if it % test_interval == 0:
        train_loss[it//test_interval] = solver.net.blobs['loss'].data
        end=time.time()
        print end-start        
        a=Error(solver.net.blobs[estimate_rgb].data.reshape(train_batch_size,3),solver.net.blobs[label].data.reshape(train_batch_size,3))
        train_average_error[it//test_interval]=a[0]
        print 'Iteration',it,'train_average_error=',train_average_error[it//test_interval]
        train_median_error[it//test_interval]=median(a[3:(train_batch_size+3)])
        print 'Iteration',it,'train_median_error=',train_median_error[it//test_interval]
        loss = 0
        for test_it in range(test_iter):
            solver.test_nets[0].forward()
            Angular=Error(solver.test_nets[0].blobs[estimate_rgb].data, solver.test_nets[0].blobs[label].data.reshape(test_batch_size,3))        
            patch_mean_error[test_it]=Angular[0]
            patch_median_error[(test_it*test_batch_size):(test_it*test_batch_size+test_batch_size-1)]=Angular[3:(test_batch_size+2)]
            patch_median_error[test_it*test_batch_size+test_batch_size-1]=Angular[test_batch_size+2]
            mean_pooling_error[test_it]=Angular[1]
            median_pooling_error[test_it]=Angular[2]
            
            per_patch_estimate[test_it]=solver.test_nets[0].blobs[estimate_rgb].data
            per_image_label[test_it]=solver.test_nets[0].blobs[label].data[1].reshape(3)            
# we can infer max or min error from mean_pooling_error or median_pooling_error 
        test_patch_mean_error[it//test_interval]=mean(patch_mean_error) 
        print '                                        val_patch_average_error=',test_patch_mean_error[it//test_interval]  
        test_patch_median_error[it//test_interval]=median(patch_median_error)
        print '                                         val_patch_median_error=',test_patch_median_error[it//test_interval]
#mean pooling
        test_mean_pooling_mean_error[it//test_interval]=mean(mean_pooling_error)
        print '                                  average_pooling_average_error=',test_mean_pooling_mean_error[it//test_interval]
        test_mean_pooling_median_error[it//test_interval]=median(mean_pooling_error)
        print '                                   average_pooling_median_error=',test_mean_pooling_median_error[it//test_interval]    
#median pooling            
        test_median_pooling_mean_error[it//test_interval]=mean(median_pooling_error)
        print '                                   median_pooling_average_error=',test_median_pooling_mean_error[it//test_interval] 
        test_median_pooling_median_error[it//test_interval]=median(median_pooling_error)
        print '                                    median_pooling_median_error=',test_median_pooling_median_error[it//test_interval]   


plt.figure(2)
plt.plot(np.arange(it//test_interval-5),train_average_error[5:(it//test_interval)],'r',train_median_error[5:(it//test_interval)],'g')
plt.xlabel('iteration')
plt.ylabel('Angular error')
plt.title('train_patch_average_error {:.5f} Red \n train_patch_median_error {:.5f} Green'.format(min(train_average_error[5:(it//test_interval)]),min(train_median_error[0:(it//test_interval)]))) 
plt.savefig(png_PATH + 'train_Angular_error.png')
np.save(png_PATH + 'train_average_error.npy',train_average_error)
np.save(png_PATH + 'train_median_error.npy',train_median_error)

plt.figure(3)
plt.plot(np.arange(it//test_interval-5),test_patch_mean_error[5:(it//test_interval)],'r',test_patch_median_error[5:(it//test_interval)],'g')
plt.xlabel('iteration')
plt.ylabel('Angular error')
plt.title('test_patch_average_error {:.5f} Red \n test_patch_median_error {:.5f} Green'.format(min(test_patch_mean_error[0:(it//test_interval)]),min(test_patch_median_error[0:(it//test_interval)])))
plt.savefig(png_PATH + 'patch_Angular_error.png') 
np.save(png_PATH + 'test_patch_mean_error.npy',test_patch_mean_error)
np.save(png_PATH + 'test_patch_median_error.npy',test_patch_median_error)

plt.figure(4)
plt.plot(np.arange(it//test_interval-5),test_mean_pooling_mean_error[5:(it//test_interval)],'r',test_mean_pooling_median_error[5:(it//test_interval)],'g')
plt.xlabel('iteration')
plt.ylabel('Angular error')
plt.title('average_pooling_average_error {:.5f} Red \n average_pooling_median_error {:.5f} Green'.format(min(test_mean_pooling_mean_error[0:(it//test_interval)]),min(test_mean_pooling_median_error[0:(it//test_interval)])))
plt.savefig(png_PATH + 'average_pooling_Angular_error.png') 
np.save(png_PATH + 'test_mean_pooling_mean_error.npy',test_mean_pooling_mean_error)
np.save(png_PATH + 'test_mean_pooling_median_error.npy',test_mean_pooling_median_error)

plt.figure(5)
plt.plot(np.arange(it//test_interval-5),test_median_pooling_mean_error[5:(it//test_interval)],'r',test_median_pooling_median_error[5:(it//test_interval)],'g')
plt.xlabel('iteration')
plt.ylabel('Angular error')
plt.title('median_pooling_average_error {:.5f} Red \n median_pooling_median_error {:.5f} Green'.format(min(test_median_pooling_mean_error[0:(it//test_interval)]),min(test_median_pooling_median_error[0:(it//test_interval)]))) 
plt.savefig(png_PATH + 'median_pooling_Angular_error.png') 
np.save(png_PATH + 'test_median_pooling_mean_error.npy',test_median_pooling_mean_error)
np.save(png_PATH + 'test_median_pooling_median_error.npy',test_median_pooling_median_error)


np.save(png_PATH+'mean_pooling_error.npy',mean_pooling_error)
np.save(png_PATH+'median_pooling_error.npy',median_pooling_error)
plt.show()  
