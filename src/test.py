import os

import numpy as np
import scipy.misc
import tensorflow as tf

from configs import *
from model import generator


def ProcessingImage(sess, filename):
    print(filename)
    image_raw = tf.gfile.FastGFile(filename, 'rb').read()
    image = tf.image.decode_png(image_raw)

    image = tf.image.resize_image_with_crop_or_pad(image, PATCH_SIZE,
                                                   PATCH_SIZE)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    downscale_size = [INPUT_SIZE, INPUT_SIZE]

    # r的值为0，1，2
    # 当r=0时，采用resize_nn；当r=1时，采用resize_area；当r=2时，采用resize_cubic
    r = sess.run(tf.random_uniform([], 0, 3, dtype=tf.int32))
    lr_image = tf.image.resize_bicubic([image], downscale_size, True)
    if r == 0:
        lr_image = tf.image.resize_nearest_neighbor([image], downscale_size,
                                                    True)
    elif r == 1:
        lr_image = tf.image.resize_area([image], downscale_size, True)
    elif r == 2:
        lr_image = tf.image.resize_bicubic([image], downscale_size, True)
    lr_image = tf.clip_by_value(lr_image, 0, 1.0)
    lr_image = tf.reshape(lr_image, [INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    save_filename1 = TEST_DATA_PATH + '/input_image.png'

    save_filename2 = TEST_DATA_PATH + '/target_image.png'

    with tf.gfile.FastGFile(save_filename1, 'wb') as f:
        _image = tf.image.convert_image_dtype(lr_image, tf.uint8)
        f.write(sess.run(tf.image.encode_png(_image)))

    with tf.gfile.FastGFile(save_filename2, 'wb') as f:
        image_ = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        f.write(sess.run(tf.image.encode_png(image_)))

    return lr_image, image


def main():
    # ========================================
    #           Create Network
    # ========================================
    test_image = tf.placeholder(
        dtype=tf.float32, shape=[1, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    sess = tf.Session()

    init = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]
    sess.run(init)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Image Super Reslution
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        inferences = generator(test_image)

    # ========================================
    #           Load Model
    # ========================================
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # ========================================
    #           Load Image
    # ========================================
    filename = os.path.join(TEST_DATA_PATH, '202598.jpg')
    input_image, target_image = ProcessingImage(sess, filename)

    input_image = sess.run(
        tf.reshape(input_image, [1, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS]))

    sr_img = sess.run(inferences, feed_dict={test_image: input_image})

    new_img = tf.reshape(sr_img, shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])

    # Save the image
    scipy.misc.toimage(
        sess.run(new_img), cmin=0.0, cmax=1.0).save(
            os.path.join(TEST_DATA_PATH, 'HR_Image.png'))

    # Compute MSE, PSNR and SSIM over tf.float32 Tensors.
    im1 = new_img
    im2 = target_image

    mse = tf.reduce_mean(tf.square(im1 - im2))

    psnr = tf.image.psnr(im1, im2, max_val=1.0)

    ssim = tf.image.ssim(im1, im2, max_val=1.0)

    message = 'mse={:5f}, '.format(sess.run(mse)) + 'psnr={:5f}, '.format(
        sess.run(psnr)) + 'ssmi={:5f}.'.format(sess.run(ssim))

    print(message)

    coord.request_stop()
    coord.join(threads=threads)


if __name__ == '__main__':
    main()
