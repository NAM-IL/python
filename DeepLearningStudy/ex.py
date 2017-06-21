'''
Created on 2017. 5. 27.

@author: KNI
'''
import os
from multiprocessing.sharedctypes import _new_value
# silence INFO logs set it to 1
# filter out WARNING set it to 2 
# silence ERROR logs (not recommended) set it to 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(tf.random_normal([1], seed=5.0)) #Variable(5.0)

hypothesis = X*W 

cost = tf.reduce_mean(tf.square(hypothesis-Y, "cost"))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# filename_queue = tf.train.string_input_producer(
# ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# writer = tf.write_file('data-01-test-score.csv', contents, name)
# reader = tf.TextLineReader()
# key, value = reader.ad(filename_queue)
# reader.restore_state(state, name)

w_val = []
cost_val = []
step_val = []

for step in range(100):
    curr_cost, curr_w = sess.run([cost, W])
    print(curr_cost, curr_w)
    sess.run(train)
    cost_val.append(curr_cost)
    w_val.append(curr_w)
    step_val.append(step)
    
plt.plot(step_val, cost_val)
# x=np.array(100)
# plt.plot(x, np.exp(-x))
plt.show()