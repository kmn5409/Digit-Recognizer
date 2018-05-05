# Plot ad hoc mnist instances
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
# load (downloaded if needed) the MNIST dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#y_train = np.array(df['label'].values)
#print(y_train)
k = 0
X_train = (train.ix[:,1:].values.astype('float32'))
y_train = train.ix[:,0].values.astype('int32')
X_test = test.values.astype('float32')
print(X_train[2])
X_train = X_train.reshape(X_train.shape[0], 28, 28)
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
seed = 7
np.random.seed(seed)
X_train/= 255
X_test/= 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes

