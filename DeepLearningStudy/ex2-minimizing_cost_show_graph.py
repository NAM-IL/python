# Lab 3 Minimizing Cost

import os
import tensorflow as tf
import time

# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm # Colormaps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

X = [1, 2, 3]
XX = [1.2, 2.1, 3.1]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
WW = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W + XX*WW

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

with tf.device('/cpu:0'):
    
    # Launch the graph in a session.
    with tf.Session() as sess:
    
        # Initializes global variables in the graph.
        sess.run(tf.global_variables_initializer())
        
        # Variables for plotting cost function
        W_history = []
        cost_history = []
        
        # Plot the softmax output for 2 dimensions for both classes
        # Plot the output in function of the weights
        # Define a vector of weights for which we want to plot the ooutput
        nb_of_zs = 150
        zs = np.linspace(-5, 10, num=nb_of_zs) # input 
        zs_1, zs_2 = np.meshgrid(zs, zs) # generate grid
        y = np.zeros((nb_of_zs, nb_of_zs, 2)) # initialize output

        # Fill the output matrix for each combination of input z's
        
        for i in range(-50,100):
            for j in range(-50,100):
                y[i,j,:] = sess.run(cost, feed_dict={W: zs_1[i,j]*0.1 , WW: zs_2[i,j]*0.1 })
          
        # Plot the cost function surfaces for both classes
        fig = plt.figure()
      
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(zs_1, zs_2, y[:,:,0], linewidth=0, cmap=cm.coolwarm)

        plt.grid()
        plt.show()
        print(y)
#  start_time = time.time()
         
#         for i in range(-50, 70):
#             curr_W = i * 0.1
#             curr_cost = 
#             W_history.append(curr_W)
#             cost_history.append(curr_cost)
#         
#         duration = time.time() - start_time
#         print(duration)
#         
#         # Show the cost function
#         plt.plot(W_history, cost_history)
        
        plt.show()
        
