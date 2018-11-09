import numpy as np
import tensorflow as tf
x= np.random.normal(loc=0,scale=1,size=[300,1])
d = x ** 2 -2 * x +1 +np.random.normal(loc=0,scale=0.5,size=[300,1])
# 10个样本，设置NO，指的是任意参数
x_net = tf.placeholder(tf.float16,shape=[10,1],name='x')
d_net = tf.placeholder(tf.float16,shape=[10,1],name='y')

#[10,1][1,1] --> [10,1] 
a = tf.Variable(np.ones([1,1])*0.5, dtype=tf.float16,name='a')
b = tf.Variable(np.ones([1])*0.5, dtype=tf.float16, name='b')

y_net = tf.matmul(x_net,a) + b
#loss 函数
loss = tf.reduce_mean(tf.square(y_net-d_net))
#梯度下降,需要减小的量loss，进度高
ts= tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 梯度下降 速度快 （优选）
ts = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#
tf.summary.FileWriter("logdir-one",sess.graph)
for itr in range(100):
    # 10 是取10个样本，根据上边设置的值确定。
    idx = np.random.randint(0,300,10)
    x_in = x[idx,:]
    d_in = d[idx,:]
    sess.run(ts,feed_dict={x_net:x_in, d_net:d_in})
print(sess.run([a.value(),b.value()]))
"""
训练集loss  测试集loss
小           小        ok
大           大      调整模型复杂度
小           大       调整模型复杂度
大           小          *
如果发现是：步长太大
"""