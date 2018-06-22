import os

import numpy as np
import scipy.misc
import tensorflow as tf

from configs import *
from model import generator


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
    filename = os.path.join(TEST_DATA_PATH, '202598.png')
    image_raw = tf.gfile.FastGFile(filename, 'rb').read()
    img = tf.image.decode_png(image_raw)
    img = tf.reshape(img, shape=[1, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    image = sess.run(img) / 255.0

    sr_img = sess.run(inferences, feed_dict={test_image: image})

    new_img = sess.run(tf.reshape(sr_img, shape=[128, 128, 3]))

    # Save the image
    scipy.misc.toimage(
        new_img, cmin=0.0, cmax=1.0).save(
            os.path.join(TEST_DATA_PATH, 'HR_Image.png'))

    SR_image = tf.image.decode_png('../data/test/HR_Image.png')
    GROUND_TRUTH = tf.image.decode_png('../data/ground_truth/202598.png')

    # Compute MSE, PSNR and SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(SR_image, tf.float32)
    im2 = tf.image.convert_image_dtype(GROUND_TRUTH, tf.float32)

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
