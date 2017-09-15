# https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import pygpu
import theano
import tensorflow as tf

print('pygpu - version: {0}'.format(pygpu.__version__))
print('theano - version: {0}'.format(theano.__version__))
print('tensorflow - version: {0}'.format(tf.__version__))

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("X_train.shape: {}".format(X_train.shape))

# create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))
    print("X_train.shape: {}".format(X_train[i].shape))


# show the plot
pyplot.show()