__Author__ = '_K_'
# -*- coding:utf-8 -*-
"""
添加了BN
改成多层感知器
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs,in_training,in_size,out_size,activation_function=None,name='layer',batch_normalization=True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.zeros([in_size,out_size]),name='W')
            tf.summary.histogram(name+'/weights',Weights)
        if not batch_normalization:
            with tf.variable_scope('biases'):
                biases = tf.Variable(tf.zeros([out_size]) + 0.01,name='b')
                tf.summary.histogram(name+'/biases',biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs,Weights) + biases
                tf.summary.histogram(name+'/Wx_plus_b_withoutBN', Wx_plus_b)
        else:
            # the bias b can be ignored since its effect will be canceled by the subsequent mean subtraction
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs,Weights)
                tf.summary.histogram(name+'/Wx_plus_b_withoutBN', Wx_plus_b)
            # excute batch normalization
            with tf.name_scope('batch_normalization'):
                with tf.name_scope('pop_mean_var'):
                    decay = 0.99
                    pop_mean = tf.Variable(tf.zeros([out_size]),trainable=False,name='pop_mean')
                    pop_var = tf.Variable(tf.ones([out_size]),trainable=False,name='pop_var')
                    shift = tf.Variable(tf.zeros([out_size]), dtype=tf.float32, name='shift')
                    scale = tf.Variable(tf.ones([out_size]), dtype=tf.float32, name='scale')
                    # in_training is a Tensor and cannot be accepted like `if in_training`. Should use tf.cond
                    def mean_var_train():
                        batch_mean, batch_var = tf.nn.moments(Wx_plus_b, axes=[0])
                        mean_train = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
                        var_train = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                        with tf.control_dependencies([mean_train,var_train]):
                            return tf.nn.batch_normalization(Wx_plus_b, batch_mean, batch_var, shift, scale, 0.001)
                    def mean_var_NOT_train():
                        return tf.nn.batch_normalization(Wx_plus_b, pop_mean, pop_var, shift, scale, 0.001)

                    Wx_plus_b = tf.cond(in_training, mean_var_train, mean_var_NOT_train)

                tf.summary.histogram(name+'/Wx_plus_b_withBN', Wx_plus_b)

        if activation_function is None:
            return Wx_plus_b
        else:
            return activation_function(Wx_plus_b)



mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

with tf.name_scope('control'):
    in_training = tf.placeholder(tf.bool,name='in_training')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x')
    y_ = tf.placeholder(tf.float32, [None, 10],name='y_')


l1 = add_layer(
    x,
    in_training,
    784,
    100,
    activation_function=tf.nn.sigmoid,
    name='hidden_layer1',
    batch_normalization=True)

l2 = add_layer(
    l1,
    in_training,
    100,
    10,
    activation_function=None,
    name='hidden_layer2',
    batch_normalization=True)

# output layer is not a fully-connected layer
with tf.name_scope('output_layer'):
    y = tf.nn.softmax(l2,name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)



with tf.name_scope('training'):
    train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)




with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./run2/",sess.graph)
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,in_training:True})
        if i % 50 == 0:
            print('iteration:',i)
            result = sess.run(merged, feed_dict={x:mnist.test.images,y_:mnist.test.labels,in_training:False})
            writer.add_summary(result, i)

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,in_training:False}))
