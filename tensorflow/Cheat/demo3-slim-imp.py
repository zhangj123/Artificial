# by cangye@hotmail.com

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


data = pd.read_csv("data/creditcard.csv")
class1 = data[data.Class==0]
class2 = data[data.Class==1]
print(len(class1))
print(len(class2))
print(np.shape(class1.values))

data1 = class1.values
data2 = class2.values
x = tf.placeholder(tf.float32, [None, 28], name="input_x")
label = tf.placeholder(tf.float32, [None, 2], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = slim.fully_connected(x, 28, activation_fn=tf.nn.relu, 
                              scope='full1', reuse=False)
net = slim.fully_connected(net, 28, activation_fn=tf.nn.relu, 
                              scope='full2', reuse=False)
y = slim.fully_connected(net, 2, activation_fn=tf.nn.sigmoid, 
                              scope='full3', reuse=False)
loss = tf.reduce_mean(tf.square(y-label))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(3000):
    idx1 = np.random.randint(284000)
    idx2 = np.random.randint(400)
    feedx = np.concatenate([data1[idx1:idx1+25, 1:29],
                            data2[idx2:idx2+25, 1:29]])
    feedy = np.zeros([50, 2])
    feedy[:25, 0] = 1
    feedy[25:, 1] = 1
    sess.run(train_step, feed_dict={x: feedx, label: feedy})
    if itr % 30 == 0:
        feedx = np.concatenate([data1[3000:3000+400, 1:29],
                                data2[:400, 1:29]])
        feedy = np.zeros([800, 2])
        feedy[:400, 0] = 1
        feedy[400:, 1] = 1
        print("step:%6d  accuracy:"%itr, 100*sess.run(accuracy, feed_dict={x: feedx,
                                        label: feedy}))
