'''
Created on Apr 9, 2017

@author: dan
'''

from __future__ import print_function

from collections import Counter
import random
import sys, os
import threading

import fire
from keras import backend as K
from keras import initializers
import keras
from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import Dense, Activation, TimeDistributed, Embedding
from keras.layers import LSTM, GRU, Dropout
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from scipy.misc import imsave
from scipy.stats._discrete_distns import logser
from sklearn.cluster import KMeans

from digits import DigitData
import image_processing
from keras_data import Manualeval
import keras_data
import numpy as np
import tensorflow as tf
import tfboard


# from sklearn.model_selection import StratifiedKFold
batch_size = 50
num_classes = 10
epochs = 20
train_size = 9000
test_size = 1000
image_size = 64
# input image dimensions
img_rows, img_cols = 64, 64
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = keras_data.get_data(distorted=False, imageSize=image_size, isValid=False, isWhole=True)
print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)
print("test:", x_test.shape, y_test.shape)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices


x_train = np.reshape(x_train, (len(x_train), 64 * 64))
x_valid = np.reshape(x_valid, (len(x_valid), 64 * 64))
x_test = np.reshape(x_test, (len(x_test), 64 * 64))



kmeans = KMeans(n_clusters=10, random_state=0).fit(x_train)
print(kmeans.labels_)
label_dic={}
for i,l in enumerate(kmeans.labels_):
    if not l in label_dic:
        label_dic[l]=np.zeros((10,1))
    label_dic[l][y_train[i]]+=1
    
#calculate misclassification
for j in label_dic.keys():
    
    cluster_label=np.argmax(label_dic[j])#true label
    wrong=0
    for l,k in enumerate(label_dic[j]):
        if l==cluster_label:#skip the right one
            continue
        wrong+=k #calculate the amount of wrong
    r=str(label_dic[j].T.tolist()[0])
    print(str(1-wrong/np.sum(label_dic[j]))+','+r)
    
#kmeans.predict(rst[900:1000])


