#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
两个向量相乘
"""
print(__doc__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = - 5,  + 5
y_min, y_max = - 5,  + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 

fig = plt.figure()
ax = fig.add_subplot(111)
p = np.c_[xx.ravel(), yy.ravel()]
d = np.array([[1.], [1.]])
Z = np.dot(p, d)
Z = Z.reshape(xx.shape)
surf = ax.contourf(xx, yy, Z)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.axis("equal")
plt.show()
