from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans
import numpy as np
centers = [[0, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)    
clt = KMeans(4)
clt.fit(X)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
Z = clt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
yp = clt.predict(X)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')
plt.axis("equal")
plt.show()