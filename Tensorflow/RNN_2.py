'''
Created on Mar 29, 2017

@author: dan
'''

from datetime import datetime
import math
import os
import time

import fire

from digits import DigitData
import image_processing
import numpy as np
import tensorflow as tf


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 64
    hidden_size = 200
    in_size = 64
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    forget_bias = 0.0
    data_type = tf.float32
    num_classes = 11

    def small(self):
        return self
    def medium(self):
        self.init_scale = 0.05
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 64
        self.hidden_size = 650
        self.in_size = 64
        self.max_epoch = 6
        self.max_max_epoch = 39
        self.keep_prob = 0.5
        self.lr_decay = 0.8
        self.batch_size = 20
        self.forget_bias = 0.0
        self.data_type = tf.float32
        self.num_classes = 11
        return self
    def large(self):
        self.init_scale = 0.04
        self.learning_rate = 1.0
        self.max_grad_norm = 10
        self.num_layers = 2
        self.num_steps = 64
        self.hidden_size = 1500
        self.in_size = 64
        self.max_epoch = 14
        self.max_max_epoch = 55
        self.keep_prob = 0.35
        self.lr_decay = 1 / 1.15
        self.batch_size = 20
        self.forget_bias = 0.0
        self.data_type = tf.float32
        self.num_classes = 11
        return self
    def custom(self, logdir,perment=False, init_scale=None, learning_rate=None, max_grad_norm=None,
                num_layers=None, num_steps=None, hidden_size=None, in_size=None, max_epoch=None,
                max_max_epoch=None, keep_prob=None, lr_decay=None, batch_size=None,
                forget_bias=None, data_type=None, num_classes=None):
        conf_new = Config()
        conf_new.init_scale = init_scale if init_scale else self.init_scale
        conf_new.learning_rate = learning_rate if learning_rate else self.learning_rate
        conf_new.max_grad_norm = max_grad_norm if max_grad_norm else self.max_grad_norm
        conf_new.num_layers = num_layers if num_layers else self.num_layers
        conf_new.num_steps = num_steps if  num_steps else self.num_steps
        conf_new.hidden_size = hidden_size if hidden_size else self.hidden_size
        conf_new.in_size = in_size if in_size else self.in_size
        conf_new.max_epoch = max_epoch if max_epoch else self.max_epoch
        conf_new.max_max_epoch = max_max_epoch if max_max_epoch else self.max_max_epoch
        conf_new.keep_prob = keep_prob if keep_prob else self.keep_prob
        conf_new.lr_decay = lr_decay if lr_decay else self.lr_decay
        conf_new.batch_size = batch_size if batch_size  else self.batch_size
        conf_new.forget_bias = forget_bias if forget_bias else self.forget_bias
        conf_new.data_type = data_type if data_type else self.data_type
        conf_new.num_classes = num_classes if num_classes else self.num_classes
        if perment:

            if not os.path.exists(logdir):
                os.makedirs(logdir)
            with open(logdir + "/config.cfg", "a") as config_file:
                attrs = vars(conf_new)
                print('\n'.join("%s: %s" % item for item in attrs.items()))
                config_file.write(str(datetime.now()))
                config_file.write("\n")
                config_file.write(', '.join("%s: %s" % item for item in attrs.items()))
                config_file.write("\n\n")
        return conf_new

class RNNInput(object):
    train_set = DigitData(subset="train")
    valid_set = DigitData(subset="validation")
    def __init__(self, config, isTrain):
        self.config = config
        self.isTrain = isTrain
        return

    def get_data(self):

        return self._data_inputs(self.isTrain, False)

    def _data_inputs(self, isTrain, isOnehot):
        if isTrain:
            dataset = self.train_set
        else:
            dataset = self.valid_set
#         batch_x, batch_y = image_processing.distorted_inputs(dataset, self.config.batch_size, self.config.num_steps, 4)
        batch_x, batch_y = image_processing.batch_inputs(dataset, self.config.batch_size, isTrain, self.config.num_steps, 4)
        batch_x = tf.image.rgb_to_grayscale(batch_x)
        batch_x = tf.reshape(batch_x, [self.config.batch_size, self.config.num_steps, self.config.in_size])
        batch_y = tf.Print(batch_y, [batch_y], 'Flat = ', summarize=self.config.in_size, first_n=1)
        batch_y = tf.cast(batch_y, tf.int32) - 0
        if isOnehot:
            batch_yy = tf.one_hot(batch_y, 10, 1, 0, -1)
            batch_yy = tf.Print(batch_yy, [batch_yy], 'Flat = ', summarize=self.config.in_size, first_n=1)
        # batch_x=tf.Print(batch_x, [batch_x],'Pixels = ',summarize=3, first_n=20)
#         batch_x = tf.cast(batch_x, tf.float32) + 1
#         batch_x = tf.cast(batch_x, tf.int32)
#         batch_x = tf.cast(batch_x, tf.float32)
        # batch_x=tf.Print(batch_x, [batch_x],'Pixels = ',summarize=3, first_n=20)
        print("get data...")
        return batch_x, batch_y

class RNNModel(object):

    RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
    RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
    RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self, config, inputs_data, isTrain=True):
        self.config = config
        self.inputs_data = inputs_data
        self.images, self.labels = inputs_data.get_data()

        self.labels = tf.Print(self.labels, [self.labels], 'Labels : ', summarize=self.config.batch_size, first_n=1)
        # self.labels=tf.Print(self.labels, [tf.argmax(self.labels,1)],'Flat1 = '+str(isTrain),summarize=100, first_n=100)
        # with tf.device("/gpu:0"):
        neural_nets = tf.contrib.rnn.MultiRNNCell(
            [self._lstm() for _ in range(self.config.num_layers)], state_is_tuple=True)
        self.init_state = neural_nets.zero_state(self.config.batch_size, self.data_type())
        output_, self._state = tf.nn.dynamic_rnn(neural_nets, self.images, dtype=self.data_type(), initial_state=self.init_state, parallel_iterations=64)
        # output = tf.transpose(output_, [1, 0, 2])
        # last = tf.gather(output, int(output.get_shape()[0]) - 1)
#         lst=[]
#         out  = tf.unstack(output_)
#         for o in out:
#             o=tf.reshape(o, [1,config.in_size*config.hidden_size])
#             lst.append(self.inference(o, 128, 32))
#         logits_1=tf.reshape(tf.stack(lst, 0),[config.batch_size,config.num_classes])
        last = output_[:, -1, :]
        softmax_w = tf.get_variable(
            "softmax_w", [self.config.hidden_size, self.config.num_classes], dtype=self.data_type())  # 300*10000
        softmax_b = tf.get_variable("softmax_b", [self.config.num_classes], dtype=self.data_type())  # 10000
        logits = tf.nn.softmax(tf.matmul(last, softmax_w) + softmax_b)
        self.labels = tf.to_int64(self.labels)
#         self._cost= cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=self.labels))


        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.labels, [-1])],  # 20*5[tensor] =>100 [list]
            [tf.ones([config.batch_size], dtype=self.data_type())])  # loss:100[tensor]
        self._cost = cost = tf.reduce_sum(loss) / config.batch_size


        tf.summary.scalar("loss", self._cost)
        logits = tf.Print(logits, [tf.arg_max(logits, 1)], 'logits : ', summarize=self.config.batch_size, first_n=1)
        corrects = tf.nn.in_top_k(logits, self.labels, 1)

        corrects = tf.Print(corrects, [corrects], 'Prediction : ', summarize=self.config.batch_size, first_n=1)
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        self._accuracy = accuracy

        if not isTrain:
            tf.summary.scalar("Validation Accuracy", accuracy)
            return
        else:
            tf.summary.scalar("Train Accuracy", accuracy)
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
#         optimizer = tf.train.RMSPropOptimizer(self._lr, self.RMSPROP_DECAY,
#                                      momentum=self.RMSPROP_MOMENTUM,
#                                      epsilon=self.RMSPROP_EPSILON)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        tf.summary.scalar('learning_rate', self._lr)
        tf.summary.scalar("batch size", self.config.batch_size)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # self._train_op = optimizer.minimize(self._cost,global_step=tf.contrib.framework.get_or_create_global_step())
        self._summary_op = tf.summary.merge_all()

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def inference(self,images, hidden1_units, hidden2_units):
        """Build the MNIST model up to where it may be used for inference.
        Args:
          images: Images placeholder, from inputs().
          hidden1_units: Size of the first hidden layer.
          hidden2_units: Size of the second hidden layer.
        Returns:
          softmax_linear: Output tensor with the computed logits.
        """
        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([64*200, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(64*200))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, 11],
                                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([11]),
                                 name='biases')
            logits = tf.matmul(hidden2, weights) + biases
        return logits
    def data_type(self):
        """tf.float32 or tf.float16"""
        return self.config.data_type


    def _lstm(self):
        """Cell unit"""
        return tf.contrib.rnn.LSTMCell(
                self.config.hidden_size, forget_bias=self.config.forget_bias, state_is_tuple=True)



    @property
    def cost(self):
        """Calculate loss"""
        return self._cost
    @property
    def lr(self):
        return self._lr
    @property
    def last_state(self):
        return self._state

    @property
    def init_op(self):
        return tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    @property
    def train_op(self):
        # self.images, self.labels = self._inputs_data.get_data()
        return self._train_op
    @property
    def accuracy(self):

        return self._accuracy

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def summary_op(self):
        return self._summary_op

    def _predict(self, inputs):
        # TODO: predict
        return
    def _predict_op(self):
        inputs = tf.placeholder(self.data_type(), [self.config.batch_size, self.config.num_steps, self.config.in_size], "TestData")
        logits, _ = self._predict(inputs)
        return tf.argmax(logits, 1)

class RNN():

    def __init__(self, config):
        self.config = config
    def train(self, session, sv, train_model, valid_model, verbose=False):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        # summary_writer = tf.summary.FileWriter("log/summary", graph=session.graph)
        try:
            step = 0
            costs = 0.0
            iters = 0
            start_time = time.time()
            # session.run(train_model.init_op)
            state = session.run(train_model.init_state)
            fetches = {
            "cost": train_model.cost,
            "final_state": train_model.last_state,
            # "embeddings": model.embeddings
            "run_op":train_model.train_op
            }
            epoch = 0
            old = 1
            while not coord.should_stop():
                epoch = step / (9000 / self.config.batch_size)

                if epoch != old:  # updte learning rate
                    lr_decay = self.config.lr_decay ** max(epoch + 1 - self.config.max_epoch, 0.0)
                    train_model.assign_lr(session, self.config.learning_rate * lr_decay)
                    old = epoch
                feed_dict = {}
                for i, (c, h) in enumerate(train_model.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
                if False:
                    run_options = tf.RunOptions(trace_level=3)
                    run_metadata = tf.RunMetadata()
                    rst = session.run(fetches,
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
                    sv.summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                else:
                    rst = session.run(fetches, feed_dict)

                loss_value = rst["cost"]
                costs += loss_value
                iters += train_model.config.num_steps
                state = rst["final_state"]
                duration = time.time() - start_time
                # Print an overview fairly often.
                if (step % 100 == 0 or step % (9000 / self.config.batch_size) == 0) and verbose:
                    print('Epoch: %d, Step: %d, loss: %.3f, perp: %.2f, lr: %3f (%.3f sec)' % (epoch, step, loss_value, np.exp(costs / iters), session.run(train_model.lr),
                                                            duration))
                    if step % (9000 / self.config.batch_size) == 0:
                        acc = session.run(train_model.accuracy)
                        print("Accuracy is:", acc)
                        valid_acc = session.run(valid_model.accuracy)
                        print("Validation is:", valid_acc)
                        summary_str = session.run(train_model.summary_op)
                        # summary_writer.add_summary(summary_str, step)
                        sv.summary_computed(session, summary_str, global_step=step)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        session.close()
class Entrance(object):
    def train(self, logdir, model, init_scale=None, learning_rate=None, max_grad_norm=None,
                    num_layers=None, num_steps=None, hidden_size=None, in_size=None, max_epoch=None,
                    max_max_epoch=None, keep_prob=None, lr_decay=None, batch_size=None,
                    forget_bias=None, data_type=None, num_classes=None):
        config = Config()
        if model == "small":
            config = Config().small()
        elif model == "medium":
            config = Config().medium()
        elif model == "large":
            config = Config().large()
        else:
            config = Config()
        # config = config.custom(logdir,in_size=64, num_steps=64, batch_size=20,num_classes=11,learning_rate=1.0,hidden_size=200,num_layers=2)
        config = config.custom(logdir,perment=True, init_scale=init_scale, learning_rate=learning_rate, max_grad_norm=max_grad_norm,
            num_layers=num_layers, num_steps=num_steps, hidden_size=hidden_size, in_size=in_size, max_epoch=max_epoch,
            max_max_epoch=max_max_epoch, keep_prob=keep_prob, lr_decay=lr_decay, batch_size=batch_size,
            forget_bias=forget_bias, data_type=data_type, num_classes=num_classes)

        with tf.Graph().as_default():

            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            valid_config = config.custom('',batch_size=20)
            with tf.name_scope("Train"):
                train_input = RNNInput(config, True)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    m = RNNModel(config, train_input, True)


            with tf.name_scope("Valid"):
                valid_input = RNNInput(valid_config, False)
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = RNNModel(valid_config, valid_input, False)



            sv = tf.train.Supervisor(logdir=logdir)
            with sv.managed_session() as session:
                rnn = RNN(config)
                rnn.train(session, sv, m, mvalid, True)

            tf.app.run()

if __name__ == "__main__":
    #train('log_test','small')
    fire.Fire(Entrance)
