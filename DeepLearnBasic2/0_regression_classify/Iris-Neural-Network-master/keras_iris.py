import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import pandas as pd

# Loading the dataset
dataset = pd.read_csv('0_regression_classify/Iris-Neural-Network-master/Iris_Dataset.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) #Â One Hot Encoding
values = list(dataset.columns.values)

y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.5,random_state=1)

model=Sequential()

model.add(Dense(16,input_shape=(4,)))
model.add(Activation("sigmoid"))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,nb_epoch=100,batch_size=1,verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))