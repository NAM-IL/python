# https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

# num_cases_per_batch: 10000
# 
# label_names:
#     0. airplane
#     1. automobile
#     2. bird
#     3. cat
#     4. deer
#     5. dog
#     6. frog
#     7. horse
#     8. ship
#     9. truck

# Simple CNN model for CIFAR-10
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import matplotlib.pyplot as plt
from scipy.misc import toimage

 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

 
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
 
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# # Create the model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
 

np.random.seed(seed)

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

 
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)


print("Accuracy: %.2f%%" % (scores[1]*100))
 
print("num_classes: %d" % (num_classes))

# create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(toimage(X_train[i]))
    print("X_train.shape: {}".format(X_train[i].shape))

# show the plot
plt.show()


def build_net(self):
    self.net = Sequential()

    self.net.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
    self.net.add(Activation('relu'))
    self.net.add(Convolution2D(32, 3, 3))
    self.net.add(Activation('relu'))
    self.net.add(MaxPooling2D(pool_size=(2, 2)))
    self.net.add(Dropout(0.25))

    self.net.add(Convolution2D(64, 3, 3, border_mode='same'))
    self.net.add(Activation('relu'))
    self.net.add(Convolution2D(64, 3, 3))
    self.net.add(Activation('relu'))
    self.net.add(MaxPooling2D(pool_size=(2, 2)))
    self.net.add(Dropout(0.25))

    self.net.add(Flatten())
    self.net.add(Dense(512))
    self.net.add(Activation('relu'))
    self.net.add(Dropout(0.5))
    self.net.add(Dense(10))
    self.net.add(Activation('softmax'))
     
     
def compile_net(self):
    self.net.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    self.net.summary()
     
     
def train(self, x_train, y_train):
    y_train = y_train.reshape(-1, 1)
    y_train = self.onehot_encode(y_train)
    self.net.fit(x_train, y_train,
                 batch_size=self.config['batch_size'],
                 nb_epoch=self.config['n_epoch'],
                 validation_split=0.01,
                 shuffle=True)
    # Save trained model
    if self.config['save_trained_model'] is True:
        self.net.save(self.config['save_trained_model_path'])
         
def predict(self, x_predict):
    if self.config['load_trained_model'] is True:
        self.net = keras.models.load_model(self.config['trained_model_path'])
        n_samples = len(x_predict)
        y_predict = self.net.predict_classes(x_predict)
        return y_predict

def predict2(x_predict):
    y_predict = model.predict(x_predict)
    return y_predict
     


y_hat = predict2(X_test)
yhat = np.argmax(y_hat, 1)

print("X_test.shape: {}".format(X_test.shape))
print("y_test.shape: {}".format(y_test.shape))
print("y_hat.shape: {}".format(y_hat.shape))

print("yhat[:10]: {}".format(yhat[:10]))
print("yhat[9]): {}".format(yhat[9]))
print("y_test[9]: {}".format(y_test[9]))
# plt.imshow(toimage(y_hat[9]))
print("y_hat[13]: {}".format(y_hat[13]))
print("y_test[13]: {}".format(y_test[13]))

print("yhat[9]: {0} - {1}".format(yhat[9],np.argmax(y_test[9])))
print("yhat[13]: {0} - {1}".format(yhat[13],np.argmax(y_test[13])))
print("yhat[5]: {0} - {1}".format(yhat[5],np.argmax(y_test[5])))

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax1.imshow(toimage(X_test[9]))
 
ax2 = fig.add_subplot(3, 1, 2)
ax2.imshow(toimage(X_test[13]))

ax3 = fig.add_subplot(3, 1, 3)
ax3.imshow(toimage(X_test[5]))

# show the plot
plt.show()
#  