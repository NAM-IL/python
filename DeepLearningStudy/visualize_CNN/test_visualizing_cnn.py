
import os
# silence INFO logs set it to 1
# filter out WARNING set it to 2 
# silence ERROR logs (not recommended) set it to 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential 
 

import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray

import tensorflow as tf
import theano  
theano.config.optimizer="None"

print('opencv - version: {0}'.format(cv2.__version__))
print('tensorflow - version: {0}'.format(tf.__version__))
print('theano - version: {0}'.format(theano.__version__))
 

# here we get rid of that added dimension and plot the image
def visualize_cat(model, cat, title=''):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)
    print(conv_cat.shape)
#     cv2.imshow(conv_cat)
    cv2.imshow(title, conv_cat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Note: matplot lib is pretty inconsistent with how it plots these weird cat arrays.
# Try running them a couple of times if the output doesn't quite match the blog post results.
def nice_cat_printer(model, cat, title=''):
    '''prints the cat as a 2d array'''
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)
 
    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print(conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
 
    print(conv_cat2.shape)
#     plt.imshow(conv_cat2)
    cv2.imshow(title, conv_cat2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
# Lets see the cat!
cat = cv2.imread('cat.png', cv2.IMREAD_COLOR)
    
 
cv2.imshow('Original', cat)

cv2.waitKey(0)
cv2.destroyAllWindows()
 
# what does the image look like?
print(cat.shape)

print(np.__version__)

#   
# # Lets create a model with 1 Convolutional layer
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
#   
#   
# visualize_cat(model, cat, '1 Convolutional layer')
#   
# # Keras expects batches of images, so we have to add a dimension to trick it into being nice
# cat_batch = np.expand_dims(cat,axis=0)
# conv_cat = model.predict(cat_batch)
#   
#   
#   
# # 10x10 Kernel ConvCat
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (10,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         10),    # x dimension of kernel
#                         input_shape=cat.shape))
#   
#   
# visualize_cat(model, cat,  '10x10 Kernel ConvCat')
#  
#      
# # Cat with 1 filter 
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
#   
# # Keras expects batches of images, so we have to add a dimension to trick it into being nice
# nice_cat_printer(model, cat, 'adding a dimension')
#   
#   
# # 15x15 kernel size
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (15,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         15),    # x dimension of kernel
#                         input_shape=cat.shape))
#   
# nice_cat_printer(model, cat, '15x15 kernel size')
#   
#   
# # Lets add a relu activation
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets add a new activation layer!
# model.add(Activation('relu'))
#   
# nice_cat_printer(model, cat, 'adding a relu activation with 1-filter layer')
#  
#   
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets add a new activation layer!
# model.add(Activation('relu'))
#   
# visualize_cat(model, cat, 'adding a relu activation with 3-filter layers')
#   
#   
#   
# # Max Pooling
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets add a new max pooling layer!
# model.add(MaxPooling2D(pool_size=(5,5)))
#   
# nice_cat_printer(model, cat, 'Max Pooling')
#   
#   
#   
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets add a new max pooling layer!
# model.add(MaxPooling2D(pool_size=(5,5)))
#   
# # nice_cat_printer(model, cat)
# visualize_cat(model, cat, 'adding MaxPooling2D with 3-filter layers')
#   
#   
# # Activation then pooling
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(5,5)))
#   
# nice_cat_printer(model, cat, 'Activation then pooling with 1-filter layers')
#  
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(5,5)))
#  
# visualize_cat(model, cat, 'Activation then pooling with 3-filter layers')
#  
#  
#  
# # Cat after the convolutional and pooling stages of LeNet
# # 1 filter in each conv layer for pretty printing
# model = Sequential()
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
#  
# nice_cat_printer(model, cat, '1 filter in each conv layer for pretty printing')
#  
#  
#  
#  
# # 3 filters in conv1, then 1 filter for pretty printing
# model = Sequential()
# model.add(Convolution2D(3,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Convolution2D(1,    # number of filter layers
#                         (3,    # y dimension of kernel (we're going for a 3x3 kernel)
#                         3),    # x dimension of kernel
#                         input_shape=cat.shape))
# # Lets activate then pool!
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#  
# nice_cat_printer(model, cat, '3 filters in conv1, then 1 filter for pretty printing')
#  
# #  
 
# 3 filters in both conv layers
model = Sequential()
model.add(Convolution2D(3,    # number of filter layers
                        (3,    # y dimension of kernel (we're going for a 3x3 kernel)
                        3),    # x dimension of kernel
                        input_shape=cat.shape))
 
 
# Lets activate then pool!
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(3,    # number of filter layers
                        (3,    # y dimension of kernel (we're going for a 3x3 kernel)
                        3),    # x dimension of kernel
                        input_shape=cat.shape))
 
# Lets activate then pool!
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
visualize_cat(model, cat, '3 filters in both conv layers')


