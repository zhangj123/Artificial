#-*- coding:utf-8 -*-

"""
本代码用于演示EM算法
cangye@hotmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
#data = np.array([1,1,0,1,0,0,1,0,1,1])
# 制作1000个样本
data = []
a_true = 0.3
b_true = 0.2
c_true = 0.8
for itr in range(1000):
    #投掷A硬币
    A = np.random.random()
    if A < a_true:
        #投掷硬币B
        B = np.random.random()
        if B < b_true:
            data.append(1)
        else:
            data.append(0)
    else:
        #投掷硬币C
        C = np.random.random()
        if C < c_true:
            data.append(1)
        else:
            data.append(0)
data = np.array(data)
plt.subplot(211)
plt.hist(data)

a, b, c = 0.4, 0.5, 0.7
for step in range(20):
    # E-step
    m1 = a * b ** data * (1 - b) ** (1 - data)
    m2 = (1 - a) * c ** data * (1 - c) ** (1 - data)
    mu = m1/(m1+m2)
    # M-step
    a = np.mean(mu)
    b = np.sum(mu * data)/np.sum(mu)
    c = np.sum((1 - mu) * data)/np.sum(1 - mu)
    print("step:%d,ture:(%f, %f, %f), pred:(%f, %f, %f)"%(step, a_true, b_true, c_true, a, b, c))

data=[]
for itr in range(1000):
    #投掷A硬币
    A = np.random.random()
    if A < a:
        #投掷硬币B
        B = np.random.random()
        if B < b:
            data.append(1)
        else:
            data.append(0)
    else:
        #投掷硬币C
        C = np.random.random()
        if C < c:
            data.append(1)
        else:
            data.append(0)
data = np.array(data)
plt.subplot(212)
plt.hist(data)
plt.show()