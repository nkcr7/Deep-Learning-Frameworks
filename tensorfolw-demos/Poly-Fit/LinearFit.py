__Author__ = '_K_'
#-*-coding:utf-8-*-


import tensorflow as tf
import numpy as np


# create 100 x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# Try to find out values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# Minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# Before starting, initialize the variables. We will 'run' this first
init = tf.initialize_all_variables()


# Launch the graph
sess = tf.Session()
sess.run(init)


# fit the line
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(W),sess.run(b))