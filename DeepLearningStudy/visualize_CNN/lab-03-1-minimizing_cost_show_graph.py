# Lab 3 Minimizing Cost

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import sklearn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('sklearn version:{0}'.format(sklearn.__version__))

tf.set_random_seed(777)  # for reproducibility

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

with tf.device('/cpu:0'):
    
    # Launch the graph in a session.
    sess = tf.Session()
    
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    
    # Variables for plotting cost function
    W_history = []
    cost_history = []
    
    start_time = time.time()
    
    for i in range(-50, 70):
        curr_W = i * 0.1
        curr_cost = sess.run(cost, feed_dict={W: curr_W})
        W_history.append(curr_W)
        cost_history.append(curr_cost)
    
    duration = time.time() - start_time
    print(duration)
    
    # Show the cost function
    plt.plot(W_history, cost_history)
    
    plt.show()
    

