'''
Created on Apr 5, 2017

@author: dan
'''

import sys

from keras.callbacks import Callback

from digits import DigitData
import image_processing
import numpy as np
import tensorflow as tf


# input image dimensions
train_set = DigitData(subset="train")
test_set = DigitData(subset="validation")
whole_set = DigitData("all")
# the data, shuffled and split between train and test sets
def get_data(distorted=False, imageSize=64, isValid=True, isWhole=False):
    train_size = 9000
    test_size = 1000
    images, labels = image_processing.batch_inputs(train_set, train_size, True, imageSize, 1, 1)
    
    imagess, labelss = image_processing.batch_inputs(test_set, test_size, True, imageSize, 1, 1)
    if isWhole:
        train_size = 10000
        image_all, labels_all = image_processing.batch_inputs(whole_set, train_size, True, imageSize, 1, 1)
        image_train = image_all
        labels_train = labels_all
        if distorted:
            images_all_d, labels_all_d = image_processing.distorted_inputs(whole_set, train_size, imageSize, 1)
            image_train = tf.concat([image_all, images_all_d], 0)
            labels_train = tf.concat([labels_all, labels_all_d], 0)
            train_size = train_size * 2
        image_train = tf.image.rgb_to_grayscale(image_train)
        image_train = tf.reshape(image_train, [train_size, imageSize, imageSize])
        image_test = labels_test = None
        
        
    else:
        if distorted:
            images_d, labels_d = image_processing.distorted_inputs(train_set, train_size, imageSize, 1)
            image_train = tf.concat([images, images_d], 0)
            labels_train = tf.concat([labels, labels_d], 0)
            image_train = tf.image.rgb_to_grayscale(image_train)
            train_size = train_size * 2
            image_train = tf.reshape(image_train, [train_size, imageSize, imageSize])
            
            images_v, labels_v = image_processing.distorted_inputs(test_set, test_size, imageSize, 1)
            image_test = tf.concat([imagess, images_v], 0)
            labels_test = tf.concat([labelss, labels_v], 0)
            image_test = tf.image.rgb_to_grayscale(image_test)
            test_size = test_size * 2
            image_test = tf.reshape(image_test, [test_size, imageSize, imageSize])
        else:
            
            labels_train = labels
            image_train = tf.image.rgb_to_grayscale(images)
            image_train = tf.reshape(image_train, [train_size, imageSize, imageSize])
            labels_test = labelss
            image_test = tf.image.rgb_to_grayscale(imagess)
            image_test = tf.reshape(image_test, [test_size, imageSize, imageSize])
            
        
        
    
    with tf.Session() as sess:
    
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        (x_train, y_train) = sess.run([image_train, labels_train])
        if not image_test == None:
            (x_test, y_test) = sess.run([image_test, labels_test])
        coord.request_stop()
        coord.join(threads)
    sess.close()
    if isWhole:
        # return shuffled list
        sf = zip(x_train, y_train)
        np.random.shuffle(sf)
        X, Y = zip(*sf)
        x_train = list(X)[0:train_size - 2000]
        x_test = list(X)[train_size - 2000:train_size]
        y_train = list(Y)[0:train_size - 2000]
        y_test = list(Y)[train_size - 2000:train_size]
        return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test)), (np.array(x_test), np.array(y_test))
    if isValid:
        return (x_train[0:train_size - 2000], y_train[0:train_size - 2000]), (x_train[train_size - 2000:train_size], y_train[train_size - 2000:train_size]), (x_test, y_test)
    else:
        return (x_train, y_train), (x_test, y_test), (x_test, y_test)
    
    
class Manualeval(Callback):
    '''
    Add test accuracy at epoch end.
    '''
    def __init__(self, model, x_test, y_test):
        super(Manualeval, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        print(logs.items())
        score = self.manual_eval()
        logs["test_acc"] = score[1]
        logs["test_loss"] = score[0]
        
        self.manual_eval()
    def manual_eval(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=2, batch_size=len(self.x_test))
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
