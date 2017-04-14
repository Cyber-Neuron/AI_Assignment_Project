'''
Created on Apr 2, 2017

@author: dan
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import NNModel
from digits import DigitData
import image_processing
import tensorflow as tf


train_set = DigitData(subset="train")
valid_set = DigitData(subset="validation")

FLAGS = tf.flags
FLAGS.batch_size = 10

def data_inputs(isTrain, isOnehot):
    if isTrain:
        dataset = train_set
    else:
        dataset = valid_set
    # batch_x, batch_y = image_processing.batch_inputs(dataset, FLAGS.batch_size, isTrain, 64, 4)
    batch_x, batch_y = image_processing.distorted_inputs(dataset, FLAGS.batch_size, 64, 4)
    batch_x = tf.image.rgb_to_grayscale(batch_x)
    batch_y = tf.Print(batch_y, [batch_y], 'Flat = ', summarize=64, first_n=1)
    batch_y = tf.cast(batch_y, tf.int32) - 0
    batch_yy = batch_y
    if isOnehot:
        batch_yy = tf.one_hot(batch_y, 11, 1, 0, -1)
        batch_yy = tf.Print(batch_yy, [batch_yy], 'Flat = ', summarize=64, first_n=1)
    print("get data...")
    return batch_x, batch_yy


def cnn(x_image):
   
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 20, 40])
    b_conv2 = bias_variable([40])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([16 * 16 * 40, 640])
    b_fc1 = bias_variable([640])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 40])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.25)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([640, 100])
    b_fc2 = bias_variable([100])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    
    W_fc21 = weight_variable([100, 11])
    b_fc21 = bias_variable([11])

    y_conv1 = tf.matmul(y_conv, W_fc21) + b_fc21
    return y_conv1, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



    
def do_eval(sess,
            op_correct,
           data_set):
    
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples_per_epoch() // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        
        true_count += sess.run(op_correct, feed_dict={})
    precision = float(true_count) / num_examples
    sess.run(tf.summary.scalar(data_set.name, precision))
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % 
          (num_examples, true_count, precision))

    
def run_training():

    with tf.Graph().as_default():
        images, labels = data_inputs(True, True)
        valid_images, valid_labels = data_inputs(False, True)
        y_conv, keep_prob = cnn(images)
        vy_conv, vkeep_prob = cnn(valid_images)
        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
#         train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
        valid_correct_prediction = tf.equal(tf.argmax(vy_conv, 1), tf.argmax(valid_labels, 1))
        train_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        valid_correct = tf.reduce_sum(tf.cast(valid_correct_prediction, tf.float32))
        
        summary = NNModel.summary()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
            sess.run(init)
    
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                
               
                
                if step % 1000 == 0:
                    do_eval(sess, valid_correct, valid_set)
                    do_eval(sess, train_correct, train_set)
                loss = sess.run([train_op, cross_entropy])
                
                
                
    
                
    
                duration = time.time() - start_time
    
               
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss[1], duration))
                    summary_str = sess.run(summary, feed_dict={})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
    
            coord.request_stop()
            coord.join(threads)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.03,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=54000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--hidden3',
        type=int,
        default=None,
        help='Number of units in hidden layer 3.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
#     parser.add_argument(
#         '--input_data_dir',
#         type=str,
#         default='/tmp/tensorflow/mnist/input_data',
#         help='Directory to put the input data.'
#     )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
