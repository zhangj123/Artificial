import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder

datas=pd.read_csv(r'F:\\学习资料\\光环国际\\code\\AIE03分类算法\\data\\creditcard.csv')
#print(datas)
X_data = datas[0:30]
label = datas['Class']
# encn = OneHotEncoder()
# label = encn.fit(label)
label=pd.DataFrame(label)
print(type(X_data))
print(type(label))

model = AdaBoostClassifier()
model.fit(X_data,label)
# 预测

