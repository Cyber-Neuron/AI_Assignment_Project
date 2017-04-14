'''
Created on Apr 3, 2017

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
from keras.layers import Dense, Activation, TimeDistributed, Embedding
from keras.layers import LSTM, GRU, Dropout
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from tensorflow.contrib.labeled_tensor import batch

from digits import DigitData
import image_processing
from keras_data import Manualeval
import keras_data
import numpy as np
import tensorflow as tf
import tfboard
from scipy.misc import imsave

batch_size = 100
num_classes = 10
epochs = 30
image_size = 64
# input image dimensions
img_rows, img_cols = 64, 64
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = keras_data.get_data(distorted=True, imageSize=image_size, isValid=False, isWhole=True)
print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)
print("test:", x_test.shape, y_test.shape)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# embedding_matrix=np.zeros((len(x_train),4096))
# for i in x_train:
#     embedding_matrix[i]=x_train[i]
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()


model.add(SimpleRNN(1000,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    # recurrent_initializer=initializers.orthogonal(),
                    activation='relu',
                    # batch_size=batch_size,
                    # input_shape=x_train.shape[1:],
                    input_shape=(64, 64),
                    # batch_input_shape=(batch_size,64,64),
                    return_sequences=False,
#                     stateful=True
                    ))
# model.add(LSTM(150, return_sequences=True,stateful=False))
# model.add(LSTM(100, return_sequences=False,stateful=False))
model.add(Dense(500, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
model.add(Dense(num_classes))
model.add(Activation("softmax"))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
plot_model(model, show_shapes=True, to_file='model_rnn_shapes.png')
# tensorbord=tfboard.TensorBoard(log_dir='logs_rnn', histogram_freq=1, write_graph=True, write_images=False)
csv_logger = CSVLogger('training_rnn.log')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.1, patience=1, min_lr=0.00001, verbose=1, cooldown=4)
me = Manualeval(model, x_test, y_test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          verbose=2,
          validation_data=(x_valid, y_valid), callbacks=[me, csv_logger, reduce_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#(_, _), (_, _), (x_test, y_test) = keras_data.get_data(isValid=False)

model2_1=Sequential()
model2_1.add(SimpleRNN(1000,
                    
                    # recurrent_initializer=initializers.orthogonal(),
                    activation='relu',
                    # batch_size=batch_size,
                    # input_shape=x_train.shape[1:],
                    input_shape=(64, 64),
                    # batch_input_shape=(batch_size,64,64),
                    return_sequences=False,
                    weights=model.layers[0].get_weights()
#                     stateful=True
                    ))
# model.add(LSTM(150, return_sequences=True,stateful=False))
# model.add(LSTM(100, return_sequences=False,stateful=False))
model2_1.add(Dense(500, activation='relu',weights=model.layers[1].get_weights()))


rst_=model2_1.predict(x_test, batch_size, verbose=1)
print(rst_.shape)
for i,features in enumerate(rst_):
    imsave(str(np.argmax(y_test,axis=-1)[i])+'_rnn_features.jpg',features.reshape(25,20))



rst = model.predict(x_test, batch_size, verbose=1)
pr_result = np.argmax(rst, axis=-1)
tr_result = np.argmax(y_test,axis=-1)#np.argmax(y_test,axis=-1)
bolrst = np.equal(tr_result, pr_result)


from scipy.misc import imsave
for i, bol in enumerate(bolrst):
    if not bol:
        # print(x_test[i].shape)
        imsave(str(i)+'_rnn_T:'+str(tr_result[i]) + '_P:'+str(pr_result[i]) + '.jpg', x_test[i])
