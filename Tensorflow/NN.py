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


# pylint: disable=missing-docstring
# Basic model parameters as external flags.
FLAGS = None
train_set = DigitData(subset="train")
valid_set = DigitData(subset="validation")



def data_inputs(isTrain, isOnehot):
    if isTrain:
        dataset = train_set
    else:
        dataset = valid_set
    batch_x, batch_y = image_processing.batch_inputs(dataset, FLAGS.batch_size, isTrain, 64, 4)
    batch_x = tf.image.rgb_to_grayscale(batch_x)
    batch_x = tf.reshape(batch_x, [FLAGS.batch_size, 64 * 64])
    batch_y = tf.Print(batch_y, [batch_y], 'Flat = ', summarize=64, first_n=1)
    batch_y = tf.cast(batch_y, tf.int32) - 0
    if isOnehot:
        batch_yy = tf.one_hot(batch_y, 10, 1, 0, -1)
        batch_yy = tf.Print(batch_yy, [batch_yy], 'Flat = ', summarize=64, first_n=1)
    print("get data...")
    return batch_x, batch_y




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
        images, labels = data_inputs(True, False)
        valid_images,valid_labels=data_inputs(False, False)
        print("Training")
        logits = NNModel.inference(images,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2,
                                 FLAGS.hidden3)
        valid_logits = NNModel.inference(valid_images,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2,
                                 FLAGS.hidden3)
        
        loss = NNModel.loss(logits, labels)
        

        train_op = NNModel.training(loss, FLAGS.learning_rate)

        eval_correct = NNModel.evaluation(logits, labels)
        valid_correct =  NNModel.evaluation(valid_logits, valid_labels)
        
        
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
    
                
    
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={})
    
                duration = time.time() - start_time
    
               
                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            
                            train_set)
                    print('Validation Data Eval:')
                    do_eval(sess,
                            valid_correct,
                            valid_set)
#                     print('Test Data Eval:')
#                     do_eval(sess,
#                             eval_correct,
#                             images_placeholder,
#                             labels_placeholder,
#                             data_inputs(False, False))
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
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
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
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
        default=100,
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
