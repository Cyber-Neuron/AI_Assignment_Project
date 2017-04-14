'''
Created on Apr 3, 2017

@author: dan
'''
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import os

class TensorBoard(Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard.
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
        write_images: whether to write model weights to visualize as
            image in Tensorboard.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,checkpoint_freq=5):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.c_batch=1
        self.c_batch_size=1 
        self.checkpoint_freq=checkpoint_freq

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.saver = tf.train.Saver()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        #w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        w_img = tf.expand_dims(w_img, -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()
        self.checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)
    def on_batch_end(self, batch, logs={}):
        self.c_batch=int(logs['batch'])
        self.c_batch_size=int(logs['size']) 
        #print(self.c_batch,self.c_batch_size)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step=epoch*self.c_batch*self.c_batch_size
        #print("get:",step)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        
        #print(tf.contrib.framework.get_or_create_global_step())
        if epoch%self.checkpoint_freq==0:
            self.saver.save(self.sess, self.checkpoint_file, global_step=step)
        print("lr:",K.get_value(self.model.optimizer.lr))
        #K.set_learning_phase(1)
        print(self.model.uses_learning_phase)
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
