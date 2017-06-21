'''
Created on 2017. 6. 10.

@author: KNI
'''

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

print('key=' + key)
print('value=' + value)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)


# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

with tf.Session() as sess:
        # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
    print("xy:" , sess.run(xy))
    
    coord.request_stop()
    coord.join(threads)