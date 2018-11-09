
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#用于训练参数的保存
saver = tf.train.Saver()
#载入保存的权值
saver.restore(sess, tf.train.latest_checkpoint('model'))

for itr in range(0):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x2d: np.reshape(batch_xs, [-1, 28, 28, 1]), label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x2d: np.reshape(mnist.test.images, [-1, 28, 28, 1]),
                                        label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model','handwriting'), global_step=itr)
for itr in tf.trainable_variables():
    print(itr.name)
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return (np.abs(x) + x)/2
def conv2d(x, kernel, bias, s=2):
    k1, k2, c1, c2 = np.shape(kernel)
    b, h, w, c = np.shape(x)
    h2 = int((h-k1)/s)+1
    w2 = int((w-k2)/s)+1
    out = np.zeros([b, h2, w2, c2])
    for itr1 in range(b):
        for itr2 in range(h2):
            for itr3 in range(w2):
                for itrc in range(c2):
                    itrh = itr2 * s
                    itrw = itr3 * s
                    out[itr1, itr2, itr3, itrc] = relu(np.sum(x[itr1, itrh:itrh+k1, itrw:itrw+k2, :] * kernel[:,:,:,itrc]) + bias[itrc])
    return out
def full(x, kernel, bias):
    sp = np.shape(x)
    if len(sp)==4:
        x = np.reshape(x, [sp[0], -1])
    return np.dot(x, kernel) + bias
def batch_norm(x, beta, gamma=[]):
    out = np.zeros_like(x)
    b, w, h, c = np.shape(x)
    if len(gamma)==0:
        gamma = np.ones_like(beta)
    for idx in range(b):
        for idf in range(c):
            frameall = x[:, :, :, idf]
            mu = np.mean(frameall)
            o = np.std(frameall)
            frame = x[idx, :, :, idf]
            frame = frame - mu
            out[idx,:,:,idf] = frame*gamma[idf]/np.sqrt(o**2+0.001) + beta[idf]
    return out
nw1, nb1, nw2, nb2, nw3, nb3, nw4, nb4 = sess.run([w1.value(), b1.value()
                     , w2.value(), b2.value()
                     , w3.value(), b3.value()
                     , w4.value(), b4.value()
                     ])
bt1, bt2, bt3 = sess.run([tf.get_default_graph().get_tensor_by_name("BatchNorm/beta:0"),
                          tf.get_default_graph().get_tensor_by_name("BatchNorm_1/beta:0"),
                          tf.get_default_graph().get_tensor_by_name("BatchNorm_2/beta:0")])

gm1,gm2 = sess.run([tf.get_default_graph().get_tensor_by_name("BatchNorm/gamma:0"),
                 tf.get_default_graph().get_tensor_by_name("BatchNorm_1/gamma:0")])
inputs = np.reshape(mnist.test.images[:100,:], [-1, 28, 28, 1])
print(np.shape(bt1))
net = conv2d(inputs, nw1, nb1)
net = batch_norm(net, bt1, gm1)
net = conv2d(net, nw2, nb2)
net = batch_norm(net, bt2, gm2)
net = conv2d(net, nw3, nb3)
net = batch_norm(net, bt3)
print("sums", np.sum(net[0,:,:,:]-sess.run(net3_b[0,:,:,:], feed_dict={x2d:inputs})))
ya = full(net, nw4, nb4)
print("sums", np.sum(ya - sess.run(y, feed_dict={x2d:inputs})))
a1 = np.argmax(ya, axis=1)
a2 = np.argmax(sess.run(y, feed_dict={x2d:inputs}), axis=1)
print(np.sum(a1==a2)/len(ya))
