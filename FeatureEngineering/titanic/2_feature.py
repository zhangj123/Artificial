# -*- coding: utf-8 -*-
# 加载相关模块和库
import io
import sys

import numpy as np
import pandas as pd

#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
"""
数据类型与分布研究
对于列名查看如下：
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
其中有的类数据类型是object，也就是字符类型数据。
统计字符数量
"""
print(__doc__)

# data_train = pd.read_csv(""""
# 数据类型与分布研究
# 对于列名查看如下：
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object
# 其中有的类数据类型是object，也就是字符类型数据。
# 这种类型数据
# 将其转换为数值型数据。

# 本文件介绍第一种方法
# 这种方式需要我们的分类算法有强大的曲面拟合能力
# """)

data_train = pd.read_csv("F:/学习资料/光环国际/code/AIE03-3DAY特征工程/all/code/a8_titanic/data/train.csv")

data = data_train.values

# 以最后一列数据而言所有类为
data1 = data[:, -1]
attr = set(data1)
print(set(attr))

# 那么数值化方法可以给定每个类一个数值
dic = dict()
for idx, itr in enumerate(attr):
    dic[itr]=idx
n_data1 = []
for itr in data1:
    n_data1.append(dic[itr])
print("原始字符数据：", data1)
print("转换后数据：", n_data1)

data = data_train.values

# 观察所有数值型数据
for idx, itr in enumerate(data_train.dtypes):
    if itr == np.object:
        print("第%d列字符类个数："%idx, len(set(data[:, idx])))

print("""
很明显一些列中包含的字符类比较多，对于这些数据而言是不适合做分类算法的，因此需要将其剔除
在一些算法中是可以容忍类似数据的
""")