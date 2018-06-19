import os
import sys

import scipy.misc
import tensorflow as tf


def visualize_samples(sess, high_imgs, gene_output, n=8, filename=None):
    img = high_imgs[0:n, :, :, :]
    img_ = gene_output[0:n, :, :, :]
    images = tf.concat([img, img_], 2)

    images = tf.concat([images[i] for i in range(n)], 0)

    image = sess.run(images)
    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(filename)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore checkpoints from {} ...".format(logdir))
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print(' Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print(' Global step: {}'.format(global_step))
        print(' Restoring...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        return global_step
    else:
        print('No checkpoint found')
        return None


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak * x)


def res_block(inputs,
              filters,
              kernel_size,
              strides=(1, 1),
              activation=tf.nn.relu,
              kernel_initializer=None):
    x = inputs

    # -----
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='SAME',
        kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(
        x,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=kernel_initializer)
    x = activation(x)

    # -----
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='SAME',
        kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(
        x,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=kernel_initializer)
    x = activation(x + inputs)

    return x