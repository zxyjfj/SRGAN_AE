import tensorflow as tf


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


def encoder(inputs, latern_dim=128):
    '''
    inputs: 输入的高分辨图像的tensor[-1, 128, 128, 3]\n
    latern_dim: code的维数
    '''
    x = inputs
    filters = 128
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    # -----
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size=5,
        strides=(1, 1),
        padding='SAME',
        kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    x = tf.nn.relu(x)
    # -----[-1, 64, 64, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 32, 32, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 16, 16, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 8, 8, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    # -----[-1, 4, 4, 128]

    # -----
    x = tf.reshape(x, shape=[-1, 4 * 4 * filters])
    x = tf.layers.dense(x, units=latern_dim, kernel_initializer=initializer)

    return x


def generator(inputs):
    '''
    inputs: 输入的低分辨图像的tensor[-1, 32, 32, 3]\n 
    '''
    x = inputs
    filters = 128
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    x = tf.layers.conv2d(
        x,
        filters / 2,
        kernel_size=3,
        strides=(1, 1),
        padding='SAME',
        kernel_initializer=initializer)

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(
        x, size=[x.shape[1] * 2, x.shape[2] * 2])
    # -----[-1, 64, 64, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(
        x, size=[x.shape[1] * 2, x.shape[2] * 2])
    # -----[-1, 128, 128, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.conv2d(
        x,
        3,
        kernel_size=3,
        strides=(1, 1),
        padding='SAME',
        kernel_initializer=initializer)
    x = tf.nn.tanh(x)

    return x


def discriminator(inputs):
    '''
    inputs: 输入的高分辨图像的tensor[-1, 128, 128, 3]\n
    '''
    x = inputs
    filters = 128
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    # -----
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size=5,
        strides=(1, 1),
        padding='SAME',
        kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    x = lrelu(x)
    # -----[-1, 64, 64, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 32, 32, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 16, 16, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, 2, 2, padding='SAME')
    # -----[-1, 8, 8, 128]

    # -----
    x = res_block(x, filters, kernel_size=3, kernel_initializer=initializer)
    # -----[-1, 4, 4, 128]

    # -----
    x = tf.reshape(x, shape=[-1, 4 * 4 * filters])
    x = tf.layers.dense(x, units=1, kernel_initializer=initializer)

    return x


def code_discriminator(inputs):
    '''
    inputs: code的tensor[-1, 128]
    '''
    x = inputs
    units = 512
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    # -----
    x = tf.layers.dense(x, units=units, kernel_initializer=initializer)
    x = lrelu(x)

    # -----
    x = tf.layers.dense(x, units=units, kernel_initializer=initializer)
    x = lrelu(x)

    # -----
    x = tf.layers.dense(x, units=1, kernel_initializer=initializer)

    return x