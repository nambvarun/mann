import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 4,
                     'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    # labels_squished = labels[:, -1, :, :]
    # preds_squished = preds[:, -1, :, :]
    #
    # squished_dim = np.prod(labels_squished.get_shape().as_list()[0:2])
    # squished_dim = np.prod(tf.shape(labels_squished)[0:2])
    # print(squished_dim)
    # labels_squished = tf.reshape(labels, [None, squished_dim, labels_squished.shape[2]])
    # preds_squished = tf.reshape(preds, [None, squished_dim, preds_squished.shape[2]])
    cross_entropy_out = tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, -1, :, :], logits=preds[:, -1, :, :])
    return tf.reduce_mean(cross_entropy_out)
    #############################


class MANN(tf.keras.Model):
    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        B = input_images.shape[0]
        K = input_images.shape[1] - 1
        N = input_images.shape[2]

        # zeros_place_holder = tf.placeholder(tf.float32, shape=[None, 1, N, N])
        zeros = tf.expand_dims(tf.zeros_like(input_labels[:, 0, :, :]), 1)
        input_labels_mod = tf.concat([input_labels[:, :K, :, :], zeros], 1)
        to_input = tf.concat([input_images, input_labels_mod], 3)
        # first_input = to_input[:, :K, :, :]
        # second_input = to_input[:, -1, :, :]

        out = self.layer1(tf.reshape(to_input, shape=[-1, K*N + N, 784 + N]))
        out = tf.nn.relu(out)
        out = self.layer2(out)
        out = tf.reshape(out, shape=[-1, K + 1, N, N])

        # first_out = self.layer1(tf.reshape(first_input, shape=[-1, K*N, 784 + N]))
        # first_out = self.layer2(first_out)
        # first_out = tf.nn.relu(first_out)
        # first_out = tf.reshape(first_out, shape=[-1, K, N, N])
        #
        # second_out = self.layer1(tf.reshape(second_input, shape=[-1, N, 784 + N]))
        # second_out = self.layer2(second_out)
        # second_out = tf.nn.relu(second_out)
        # second_out = tf.reshape(second_out, shape=[-1, 1, N, N])
        #
        # out = tf.concat([first_out, second_out], 1)
        #############################
        return out


ims = tf.placeholder(tf.float32, shape=(None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.0001)
optimizer_step = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(-1, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
