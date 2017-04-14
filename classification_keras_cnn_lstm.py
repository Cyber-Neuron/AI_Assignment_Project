'''
Created on Apr 3, 2017

@author: dan
'''
import sys

from keras import backend as K
import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

from digits import DigitData
import image_processing
from keras_data import Manualeval
import keras_data
import numpy as np
import tensorflow as tf
import tfboard
from scipy.misc import imsave

batch_size = 50
num_classes = 10
epochs = 20
image_size = 64

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = keras_data.get_data(distorted=True, imageSize=image_size, isValid=False, isWhole=True)
print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)
print("test:", x_test.shape, y_test.shape)



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()

model.add(Conv1D(40, 3,
                 activation='relu', input_shape=(64, 64)))
model.add(Conv1D(60, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(1000))
model.add(Dense(num_classes, kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
plot_model(model, show_shapes=True, to_file='model_cnn-lstm_shapes.png')
print('Train...')
# tensorbord=tfboard.TensorBoard(log_dir='logs_cnn_lstm', histogram_freq=1, write_graph=True, write_images=False)
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('training_cnn_lstm.log')
me = Manualeval(model, x_test, y_test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          verbose=2,
          epochs=epochs,
          validation_data=(x_valid, y_valid), callbacks=[me, csv_logger, reduce_lr])
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
#(_, _), (_, _), (x_test, y_test) = keras_data.get_data(isValid=False)

model2_1 = Sequential()

model2_1.add(Conv1D(40, 3,
                 activation='relu', input_shape=(64, 64),weights=model.layers[0].get_weights()))
model2_1.add(Conv1D(60, 5, activation='relu',weights=model.layers[1].get_weights()))
model2_1.add(MaxPooling1D(pool_size=2,weights=model.layers[2].get_weights()))
model2_1.add(LSTM(1000,weights=model.layers[3].get_weights()))
rst_=model2_1.predict(x_test, batch_size, verbose=1)
print(rst_.shape)
for i,features in enumerate(rst_):
    imsave(str(np.argmax(y_test,axis=-1)[i])+'_cnn_lstm_features.jpg',features.reshape(25,40))


rst = model.predict(x_test, batch_size, verbose=1)
pr_result = np.argmax(rst, axis=-1)
tr_result = np.argmax(y_test,axis=-1)#np.argmax(y_test,axis=-1)
bolrst = np.equal(tr_result, pr_result)


for i, bol in enumerate(bolrst):
    if not bol:
        # print(x_test[i].shape)
        imsave(str(i)+'_cnnlstm_T:'+str(tr_result[i]) + '_P:'+str(pr_result[i]) + '.jpg', x_test[i])
