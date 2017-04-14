'''
Created on Apr 2, 2017

@author: dan
'''


import sys

from keras import backend as K, regularizers
import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

from digits import DigitData
import image_processing
from keras_data import Manualeval
import keras_data
import numpy as np
import tensorflow as tf
import tfboard
import PIL
from scipy.misc import imsave

batch_size = 70
num_classes = 10
epochs = 20

image_size = 64
# input image dimensions
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
model.add(Conv1D(40, kernel_size=3,
                 activation='relu',
                 input_shape=(64, 64)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(60, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
plot_model(model, show_shapes=True, to_file='model_cnn_shapes.png')
# to decay the learning rate from epoch 34
# tensorbord = tfboard.TensorBoard(log_dir='logs_cnn', histogram_freq=1, write_graph=True, write_images=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.1, patience=1, min_lr=0.00001, verbose=1, cooldown=4)
csv_logger = CSVLogger('training_cnn.log')
me = Manualeval(model, x_test, y_test)
model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
        epochs=epochs,
#           epochs=3,
          verbose=2,
          validation_data=(x_valid, y_valid)
        , callbacks=[me, reduce_lr, csv_logger]
          )
score = model.evaluate(x_test, y_test, verbose=1)


#(_, _), (_, _), (x_test, y_test) = keras_data.get_data(isValid=False)

model2_1=Sequential()
model2_1.add(Conv1D(40, kernel_size=3,
                 activation='relu',
                 input_shape=(64, 64),weights=model.layers[0].get_weights()))
model2_1.add(MaxPooling1D(pool_size=2,weights=model.layers[1].get_weights()))
rst_=model2_1.predict(x_test, batch_size, verbose=1)
print(rst_.shape)
for i,features in enumerate(rst_):
    imsave(str(np.argmax(y_test,axis=-1)[i])+'_cnn_features.jpg',features.reshape(31,40))



rst = model.predict(x_test, batch_size, verbose=1)
pr_result = np.argmax(rst, axis=-1)
tr_result = np.argmax(y_test,axis=-1)
bolrst = np.equal(tr_result, pr_result)



for i, bol in enumerate(bolrst):
    if not bol:
        # print(x_test[i].shape)
        imsave(str(i)+'_cnn_T:'+str(tr_result[i]) + '_P:'+str(pr_result[i]) + '.jpg', x_test[i])
