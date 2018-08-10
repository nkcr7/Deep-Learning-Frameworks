import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
This is a test program, where a network of 2 hidden layers is used to fit the curve.
Batch Normalization is implemented.
Basic Tensorflow and Tensorboard APIs are used.
"""


def add_layer(inputs,in_training,in_size,out_size,activation_function=None,name='layer',batch_normalization=True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.histogram_summary(name+'/weights',Weights)
        if not batch_normalization:
            with tf.variable_scope('biases'):
                biases = tf.Variable(tf.zeros([1,out_size]) + 0.01,name='b')
                tf.histogram_summary(name+'/biases',biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs,Weights) + biases
                tf.histogram_summary(name+'/Wx_plus_b_withoutBN', Wx_plus_b)
        else:
            # the bias b can be ignored since its effect will be canceled by the subsequent mean subtraction
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs,Weights)
                tf.histogram_summary(name+'/Wx_plus_b_withoutBN', Wx_plus_b)
            # excute batch normalization
            with tf.name_scope('batch_normalization'):
                with tf.name_scope('pop_mean_var'):
                    decay = 0.99
                    pop_mean = tf.Variable(tf.zeros([out_size]),trainable=False,name='pop_mean')
                    pop_var = tf.Variable(tf.ones([out_size]),trainable=False,name='pop_var')
                    shift = tf.Variable(tf.zeros([out_size]), dtype=tf.float32, name='shift')
                    scale = tf.Variable(tf.ones([out_size]), dtype=tf.float32, name='scale')
                    tf.histogram_summary(name + '/shift', shift)
                    tf.histogram_summary(name + '/scale', scale)

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

                tf.histogram_summary(name+'/Wx_plus_b_withBN', Wx_plus_b)

        if activation_function == None:
            return Wx_plus_b
        else:
            return activation_function(Wx_plus_b)


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

with tf.name_scope('control'):
    in_training = tf.placeholder(tf.bool,name='in_training')

l1 = add_layer(
    xs,
    in_training,
    1,
    10,
    activation_function=tf.nn.relu,
    name='hidden_layer1',
    batch_normalization=True)

l2 = add_layer(
    l1,
    in_training,
    10,
    10,
    activation_function=tf.nn.relu,
    name='hidden_layer2',
    batch_normalization=True)

predict = add_layer(
    l2,
    in_training,
    10,
    1,
    activation_function=None,
    name='output_layer',
    batch_normalization=True)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(ys - predict))
    tf.scalar_summary('loss', loss)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.9)
    train = optimizer.minimize(loss)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./run4",sess.graph)
    for i in range(3000):
        sess.run(train,feed_dict={xs:x_data,ys:y_data,in_training:True})
        if i % 100 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data,in_training:False}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            predict_value = sess.run(predict,feed_dict={xs:x_data,in_training:False})
            lines = ax.plot(x_data,predict_value,'r-',lw = 5)
            plt.pause(0.1)
            predict_single1 = sess.run(predict,feed_dict={xs:[x_data[1]],in_training:False})
            predict_single2 = sess.run(predict,feed_dict={xs:[x_data[150]],in_training:False})
            print('predict:',predict_single1,predict_single2)
            result = sess.run(merged,feed_dict={xs:x_data,ys:y_data,in_training:False})
            writer.add_summary(result,i)

