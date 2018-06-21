import os

import numpy as np
import scipy.misc
import tensorflow as tf
from skimage import io, measure

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

    SR_image = io.imread('../data/test/HR_Image.png')
    GROUND_TRUTH = io.imread('../data/ground_truth/202598.png')

    # MSE
    mse = measure.compare_mse(GROUND_TRUTH, SR_image)

    # PSNR
    psnr = measure.compare_psnr(GROUND_TRUTH, SR_image, data_range=255)

    # SSMI
    ssmi = measure.compare_ssim(GROUND_TRUTH, SR_image, multichannel=True)

    message = 'mse={:5f}, '.format(mse) + 'psnr={:5f}, '.format(
        psnr) + 'ssmi={:5f}.'.format(ssmi)

    print(message)

    coord.request_stop()
    coord.join(threads=threads)


if __name__ == '__main__':
    main()
