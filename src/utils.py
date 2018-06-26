import os
import sys

import scipy.misc
import tensorflow as tf

from configs import *


def batch_queue_for_training(data_path):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(os.path.join(data_path, '*.jpg')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_jpeg(image_file, NUM_CHENNELS)

    patch = tf.image.resize_image_with_crop_or_pad(patch, PATCH_SIZE,
                                                   PATCH_SIZE)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    # 将图像的数据格式转换为tf.float32,范围是[0, 1)
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    # 随机翻转
    high_res_patch = tf.image.random_flip_left_right(patch)

    # 高斯模糊

    downscale_size = [INPUT_SIZE, INPUT_SIZE]

    def resize_nn():
        return tf.image.resize_nearest_neighbor([high_res_patch],
                                                downscale_size, True)

    def resize_area():
        return tf.image.resize_area([high_res_patch], downscale_size, True)

    def resize_cubic():
        return tf.image.resize_bicubic([high_res_patch], downscale_size, True)

    # r的值为0，1，2
    # 当r=0时，采用resize_nn；当r=1时，采用resize_area；当r=2时，采用resize_cubic
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case(
        {
            tf.equal(r, 0): resize_nn,
            tf.equal(r, 1): resize_area
        },
        default=resize_cubic)[0]

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # 确保图片的像素点的值在0-1.0范围内
    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.shuffle_batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=NUM_PROCESS_THREADS,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE,
        min_after_dequeue=MIN_QUEUE_EXAMPLES)

    return low_res_batch, high_res_batch


def batch_queue_for_testing(data_path):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(os.path.join(data_path, '*.jpg')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_jpeg(image_file, NUM_CHENNELS)

    patch = tf.image.resize_image_with_crop_or_pad(patch, PATCH_SIZE,
                                                   PATCH_SIZE)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    # 将图像的数据格式转换为tf.float32,范围是[0, 1)
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    # 随机翻转
    high_res_patch = tf.image.random_flip_left_right(patch)

    # 高斯模糊

    downscale_size = [INPUT_SIZE, INPUT_SIZE]

    def resize_nn():
        return tf.image.resize_nearest_neighbor([high_res_patch],
                                                downscale_size, True)

    def resize_area():
        return tf.image.resize_area([high_res_patch], downscale_size, True)

    def resize_cubic():
        return tf.image.resize_bicubic([high_res_patch], downscale_size, True)

    # r的值为0，1，2
    # 当r=0时，采用resize_nn；当r=1时，采用resize_area；当r=2时，采用resize_cubic
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case(
        {
            tf.equal(r, 0): resize_nn,
            tf.equal(r, 1): resize_area
        },
        default=resize_cubic)[0]

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # 确保图片的像素点的值在0-1.0范围内
    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE)

    return low_res_batch, high_res_batch


def visualize_samples(sess,
                      high_imgs,
                      gene_output,
                      interpolation,
                      n=5,
                      filename=None):
    '''
    结果可视化
    '''
    img = high_imgs[0:n, :, :, :]
    img_ = gene_output[0:n, :, :, :]
    _img = interpolation[0:n, :, :, :]
    images = tf.concat([img, _img, img_], 2)

    images = tf.concat([images[i] for i in range(n)], 0)

    image = sess.run(images)
    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(filename)
    print(' Done.')


def load(saver, sess, logdir):
    '''
    加载模型
    '''
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
    '''
    保存模型
    '''
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
