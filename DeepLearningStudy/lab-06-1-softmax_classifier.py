# Lab 6 Softmax Classifier
import os
# silence INFO logs set it to 1
# filter out WARNING set it to 2 
# silence ERROR logs (not recommended) set it to 3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
with tf.name_scope("input") as scope:
 X = tf.placeholder("float", [None, 4])

with tf.name_scope("layer1") as scope:
 Y = tf.placeholder("float", [None, 3])

nb_classes = 3

with tf.name_scope("weight") as scope:
 W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')

with tf.name_scope("bias") as scope:
 b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

with tf.name_scope("cost") as scope:
# Cross entropy cost/loss
 cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

with tf.name_scope("train") as scope:    
 optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


w_hist = tf.summary.histogram("weight", W) 
b_hist = tf.summary.histogram("bias", b) 
y_hist = tf.summary.histogram("y", Y) 
cost_hist = tf.summary.histogram("cost", cost) 
# optimizer_hist = tf.summary.histogram("train", optimizer) 


# Launch graph
with tf.Session() as sess:
 
    merged = tf.summary.merge_all()
    writer =tf.summary.FileWriter("./board/Lab_6_Softmax", sess.graph)
    
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary,step)
    print('--------------')

    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    print('--------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    print('--------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    print('--------------')

    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))

'''
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]] [1]
--------------
[[ 0.93119204  0.06290206  0.0059059 ]] [0]
--------------
[[  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [2]
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]
 [  9.31192040e-01   6.29020557e-02   5.90589503e-03]
 [  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [1 0 2]
'''
