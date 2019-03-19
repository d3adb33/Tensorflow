import tensorflow as tf
import numpy as np

x = tf.Variable(tf.random_uniform(shape=[2], minval = -2.0, maxval=2.0), dtype= tf.float32)
function = 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(function)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(100000):
        train.run()
        if it % 100 == 0:
            print(sess.run(x))