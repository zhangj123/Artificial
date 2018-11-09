
# by cangye@hotmail.com
# TensorFlow入门实例

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets("data/", one_hot=True)

#卷积函数
def conv2d_layer(input_tensor, size=1, feature=128, name='conv1d'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel', 
                                  (size, size, shape[-1], feature), 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))#初始化值很重要，不好的初始化值比如文章中的初始化值会使得迭代收敛极为缓慢。
        b = tf.get_variable('b', [feature], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 2, 2, 1], padding='VALID') + b
    return tf.nn.relu(out), kernel, b
#全链接函数
def full_layer(input_tensor, out_dim, name='full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return out, W, b


with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")
x2d = tf.placeholder(tf.float32, [None,28,28,1])
net1, w1, b1 = conv2d_layer(x2d, size=4, feature=64, name='conv1')
net1_b = slim.batch_norm(net1, scale=True)
#print(net.get_shape().as_list())
net2, w2, b2 = conv2d_layer(net1_b, size=4, feature=64, name='conv2')
net2_b = slim.batch_norm(net2, scale=True)
#print(net.get_shape().as_list())
net3, w3, b3 = conv2d_layer(net2_b, size=4, feature=64, name='conv3')
net3_b = slim.batch_norm(net3)
#print(net.get_shape().as_list())
#flatten层，用于将三维的图形数据展开成一维数据，用于全链接层
net4 = tf.contrib.layers.flatten(net3_b)
y, w4, b4=full_layer(net4, 10, name='full')

with tf.variable_scope("loss"):
    #定义loss函数
    ce=tf.square(label-tf.nn.sigmoid(y))
    loss = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
   
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#用于训练参数的保存
saver = tf.train.Saver()
#载入保存的权值
saver.restore(sess, tf.train.latest_checkpoint('model'))

for itr in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x2d: np.reshape(batch_xs, [-1, 28, 28, 1]), label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x2d: np.reshape(mnist.test.images, [-1, 28, 28, 1]),
                                        label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model','handwriting'), global_step=itr)