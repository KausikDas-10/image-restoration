"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
# import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import os
import tensorflow as tf
import PIL
import scipy.stats as st
import sys
import imageio
import cv2
# FLAGS = tf.app.flags.FLAGS
FLAGS = tf.compat.v1.flags

def transform(images):
  return np.array(images)/127.5 - 1.
def inverse_transform(images):
  return (images+1.)/2
def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  
  """
  # import pdb 
  # pdb.set_trace()
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))+ glob.glob(os.path.join(data_dir, "*.jpeg"))
  return data

def imread(path, is_grayscale=False):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  # import pdb 
  # pdb.set_trace()
  if is_grayscale:
    return cv2.resize(imageio.imread(path, flatten=True), (1024,632)).astype(np.float_)
    # return imageio.imread(path, flatten=True).astype(np.float_)
  else:
    print('path: ', path)
    return cv2.resize(imageio.imread(path), (1024,632)).astype(np.float_)
    # return imageio.imread(path).astype(np.float_)

    
def imsave(image, path):
  # import pdb 
  # pdb.set_trace()
  # imsaved = (inverse_transform(image)).astype(np.float)
  imsaved = (inverse_transform(image)).astype(np.uint8)
  return imageio.imsave(path, imsaved)

def get_image(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  # image = image.resize((1024, 632))
  # w, h = image.shape
  # if w > 1500 or h > 1500:
  #   image = image.resize((w/2, h/2))
  #   print('image resize !!')
  # import pdb 
  # pdb.set_trace()
  #return transform(image)
  return image/255
def get_lable(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255.
def imsave_lable(image, path):
  image = (image*255).astype(np.uint8)
  image = Image.fromarray(image).convert('RGB')
  return imageio.imsave(path, image)

def loss_gradient_difference(true, generated):
   true_x_shifted_right = true[:,1:,:,:]
   true_x_shifted_left = true[:,:-1,:,:]
   true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

   generated_x_shifted_right = generated[:,1:,:,:]
   generated_x_shifted_left = generated[:,:-1,:,:]
   generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

   loss_x_gradient = tf.reduce_mean(tf.square(true_x_gradient - generated_x_gradient))

   true_y_shifted_right = true[:,:,1:,:]
   true_y_shifted_left = true[:,:,:-1,:]
   true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

   generated_y_shifted_right = generated[:,:,1:,:]
   generated_y_shifted_left = generated[:,:,:-1,:]
   generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
   loss_y_gradient = tf.reduce_mean(tf.square(true_y_gradient - generated_y_gradient))

   loss = loss_x_gradient + loss_y_gradient
   return loss

def gredient(x):
  # _,h,w,_=x.shape
  # g_x = np.zeros(x.shape)
  # g_y = np.zeros(x.shape)
  g_x = x[:,0:-1,:,:]-x[:,1:,:,:]
  # g_x[:,-1,:,:] = x[:,-1,:,:]
  g_y = x[:,:,0:-1,:]-x[:,:,1:,:]
  # g_y[:,:,-1,:] = x[:,:,-1,:]
  g = tf.reduce_mean(tf.abs(g_x))+tf.reduce_mean(tf.abs(g_y))
  return g


#def random_crop_and_flip(batch_data, padding_size):
def random_crop_and_flip_3(batch_data, x_offset, y_offset, w_size,h_size):
    
    cropped_batch = np.zeros(len(batch_data) * w_size * h_size * 3).reshape(
                 len(batch_data), w_size, h_size, 3)
    
    for i in range(len(batch_data)):
        #x_offset = np.random.randint(low=0, height=2*padding_size, size=1)[0]
        #x_offset = np.random.randint(low=0, height=2 * padding_size, size=1)[0]
        #import pdb  
        #pdb.set_trace()
        cropped_batch[i, :,:,:] = batch_data[i, x_offset:x_offset+w_size,y_offset:y_offset+h_size,:]
        #cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        
    return cropped_batch

def random_crop_and_flip_1(batch_data, x_offset, y_offset,w_size,h_size):
    
    cropped_batch = np.zeros(len(batch_data) * w_size * h_size * 1).reshape(
                 len(batch_data), w_size, h_size, 1)
    
    for i in range(len(batch_data)):
        #x_offset = np.random.randint(low=0, height=2*padding_size, size=1)[0]
        #x_offset = np.random.randint(low=0, height=2 * padding_size, size=1)[0]
        #import pdb  
        #pdb.set_trace()
        cropped_batch[i, :,:,:] = batch_data[i, x_offset:x_offset+w_size,y_offset:y_offset+h_size,:]
        #cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        
    return cropped_batch

def tensor_random_crop_and_flip_3(batch_data, batch_size,x_offset, y_offset, w_size,h_size):
    # import pdb  
    # pdb.set_trace()  
    # cropped_batch = tf.zeros([1, 64, 64, 3])  
    # cropped_batch = tf.zeros([batch_size, w_size, h_size, 3])
    
    # for i in range(batch_size):
        #x_offset = np.random.randint(low=0, height=2*padding_size, size=1)[0]
        #x_offset = np.random.randint(low=0, height=2 * padding_size, size=1)[0]
        #import pdb  
        #pdb.set_trace()cropped_batch[i, :,:,:] = 
        
        #cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        
    return batch_data[:, x_offset:x_offset+w_size,y_offset:y_offset+h_size,:]

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter


def MaxMinNormalization(x):
  x = (x - x.min()) / (x.max() - x.min())
  return x