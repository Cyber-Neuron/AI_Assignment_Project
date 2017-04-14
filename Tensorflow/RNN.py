'''
Created on Mar 27, 2017

@author: dan
'''
import time

from tensorflow.contrib import rnn

from digits import DigitData
import image_processing as ip
import numpy as np
import tensorflow as tf
import copy
from tensorflow.contrib.labeled_tensor import placeholder

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = ''
# Parameters
training_iters = 10000
batch_size = 20
display_step = 10

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 300  # hidden layer num of features
num_layers = 2
n_classes = 10  # MNIST total classes (0-9 digits)

FLAGS = tf.app.flags.FLAGS
FLAGS.train_dir = "data"
FLAGS.eval_dir = "data"
final_state = None  # RNN memory
train_set = DigitData(subset="train")
valid_set = DigitData(subset="validation")
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
MOVING_AVERAGE_DECAY=0.9999
initial_learning_rate=0.1
learning_rate_decay_factor=0.16

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
def data_inputs(isTrain, isOneHot):
    if isTrain:
        dataset = train_set
    else:
        dataset = valid_set
    batch_x, batch_y = ip.distorted_inputs(dataset, batch_size, 4)
    batch_x = tf.image.rgb_to_grayscale(batch_x)
    batch_x = tf.reshape(batch_x, [batch_size, 28, 28])
    # batch_y=tf.to_float(batch_y, name='ToFloat')
    batch_yy = batch_y
    if isOneHot:
        batch_yy = tf.one_hot(batch_y, 10, 1, 0, -1)
       
#         sparse_labels = tf.reshape(batch_y, [batch_size, 1])
#         indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
#         concated = tf.concat(axis=1, values=[indices, sparse_labels])
#         num_classes = 10
#         dense_labels = tf.sparse_to_dense(concated,
#                                           [batch_x.get_shape()[0], num_classes],
#                                           1.0, 0.0)
    print("get data...")
    return batch_x, batch_yy
# tf Graph input

def run_training():
    
    with tf.Graph().as_default():
        global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = (train_set.num_examples_per_epoch() /
                                 batch_size)
        decay_steps = int(num_batches_per_epoch * 30.0)
        lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
        optimizer = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)
        variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
       
#         images,labels=mnist.train.next_batch(batch_size)
#         images=tf.reshape(images, [batch_size, 28, 28])
#         labels= tf.one_hot(labels, 10, 1, 0, -1)
        images, labels = data_inputs(True, True)
        
        neual_nets = tf.contrib.rnn.MultiRNNCell(
            [lstm() for _ in range(num_layers)], state_is_tuple=True)
        init_state=neual_nets.zero_state(batch_size, tf.float32)
        output_, state = tf.nn.dynamic_rnn(neual_nets, images, dtype=tf.float32,initial_state=init_state)
        final_state=state
        output = tf.transpose(output_, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # outputs = tf.reshape(tf.concat(axis=1, values=output), [-1, n_hidden])  # 100*300
        softmax_w = tf.get_variable(
            "softmax_w", [n_hidden, n_classes], dtype=tf.float32)  # 300*10000
        softmax_b = tf.get_variable("softmax_b", [n_classes], dtype=tf.float32)  # 10000
        logits = tf.nn.softmax(tf.matmul(last, softmax_w) + softmax_b)  
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        
        grads = optimizer.compute_gradients(cost)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        apply_gradient_op=optimizer.apply_gradients(grads, global_step=global_step)
        variables_averages_op = variable_averages.apply(variables_to_average)
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('global_step', global_step)

        
        #train_op_2 = optimizer.minimize(cost)
        train_mistakes = tf.not_equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        train_accuracy_op = tf.reduce_mean(tf.cast(train_mistakes, tf.float32))
        tf.summary.scalar('train_accuracy', train_accuracy_op)
        ######################################################
        tf.get_variable_scope().reuse_variables()
        valid_images, valid_labels = data_inputs(False, True)
        valid_output_, _ = tf.nn.dynamic_rnn(neual_nets, valid_images, dtype=tf.float32)
        valid_output = tf.transpose(valid_output_, [1, 0, 2])
        valid_last = tf.gather(valid_output, int(valid_output.get_shape()[0]) - 1)
        valid_logits = tf.nn.softmax(tf.matmul(valid_last, softmax_w) + softmax_b) 
        valid_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits, labels=valid_labels))
        tf.summary.scalar('valid loss', valid_cost)
        valid_mistakes = tf.not_equal(tf.argmax(valid_labels, 1), tf.argmax(valid_logits, 1))
        valid_accuracy_op = tf.reduce_mean(tf.cast(valid_mistakes, tf.float32))
        tf.summary.scalar('valid_accuracy', valid_accuracy_op)
        
        
        
        
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar("batch size", batch_size)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=sess.graph)
        try:
            step = 0
            start_time = time.time()
            state=sess.run(init_state)
            while not coord.should_stop():
                
                feed_dict = {}
                for i, (c, h) in enumerate(init_state):
                        print(type(c),c.shape)
                        feed_dict[c] = state[i].c
                        feed_dict[h] = state[i].h
                _, loss_value,state= sess.run([train_op, cost,final_state],feed_dict)
    
                duration = time.time() - start_time
    
                # Print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                               duration))
                    acc = sess.run(train_accuracy_op)
                    print("Accuracy is:", acc)
                    valid_acc = sess.run(valid_accuracy_op)
                    print("Validation is:", valid_acc)
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
    
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
# Define weights
def lstm():
    return tf.contrib.rnn.LSTMCell(
                n_hidden, forget_bias=0.0, state_is_tuple=True)



def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)
