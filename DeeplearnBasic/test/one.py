import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float16,shape=[4,4])
#用于接受外界的值，举证（没有初始化值，需要从外界接受）
#张量,constant常亮
# a1 = tf.constant(np.ones([4,4]))
# a2 = tf.constant(np.ones([4,4]))
#变量
#float32,对精度要求不高。16位用到机器学习。64时用来做科学计算的。
a1 = tf.Variable(np.ones([4,4]),dtype=tf.float16)
a2 = tf.Variable(np.ones([4,4]),dtype=tf.float16)

c1= tf.matmul(a1,a2)
#y = a*x +b
c1 = tf.matmul(x,a2)
sess = tf.Session()

#变量 需要初始化,分配内存空间
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(c1,feed_dict={x:np.ones([4,4])}))
print(sess.run(c1))