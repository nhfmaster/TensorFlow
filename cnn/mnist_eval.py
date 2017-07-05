import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from cnn import cnn_mnist_inference
from cnn import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [5000, cnn_mnist_inference.IMAGE_SIZE, cnn_mnist_inference.IMAGE_SIZE,
                            cnn_mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, cnn_mnist_inference.NUM_LABELS], name='y-input')
        # x = tf.placeholder(tf.float32, [None, cnn_mnist_inference.IMAGE_SIZE], name='x-input')
        # y_ = tf.placeholder(tf.float32, [None, cnn_mnist_inference.NUM_LABELS], name='y-input')
        xs = mnist.validation.images
        ys = mnist.validation.labels
        reshaped_xs = np.reshape(xs, (5000, cnn_mnist_inference.IMAGE_SIZE, cnn_mnist_inference.IMAGE_SIZE,
                                      cnn_mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: ys}
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        y = cnn_mnist_inference.inference(x, None, regularizer)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                    time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
