import os

import scipy.misc
import tensorflow as tf

from configs import *
from model import generator
from utils import batch_queue_for_testing, load


def main():
    lr_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    hr_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        inferences = generator(lr_holders)

    low_res_batch, high_res_batch = batch_queue_for_testing(TEST_DATA_PATH)

    # 初始化tensorflow
    sess = tf.Session()
    init = [
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ]
    sess.run(init)

    # the saver will restore all model's variables during training
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_CKPT_TO_KEEP)
    try:
        saved_global_step = load(saver, sess, CHECKPOINTS_PATH)
        if saved_global_step is None:
            saved_global_step = 0
    except:
        raise ValueError(
            "You have changed the model, Please Delete CheckPoints!")

    tf.train.start_queue_runners(sess=sess)

    low_res_images, high_res_images = sess.run([low_res_batch, high_res_batch])

    feed_dict = {lr_holders: low_res_images, hr_holders: high_res_images}

    hr_imgs = sess.run(inferences, feed_dict=feed_dict)

    total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0

    count = len([
        name for name in os.listdir(TEST_DATA_PATH)
        if os.path.isfile(os.path.join(TEST_DATA_PATH, name))
    ])

    for i in range(count):
        # 生成的图片
        new_image = tf.reshape(hr_imgs[i],
                               [PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])

        # 高分辨率图片
        target_image = tf.reshape(high_res_images[i],
                                  [PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])

        # 三次样条插值生成的图片
        bicubic_image = tf.image.resize_bicubic([lr_holders[i]],
                                                [PATCH_SIZE, PATCH_SIZE])

        bicubic_image = tf.reshape(
            bicubic_image, shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])

        img = tf.concat([target_image, bicubic_image, new_image], 1)
        # Save the image
        scipy.misc.toimage(
            sess.run(img, feed_dict=feed_dict), cmin=0.0, cmax=1.0).save(
                os.path.join(TEST_GENERATOR_TRUTH,
                             'HR_Image_{}.png'.format(i)))

        # Compute MSE, PSNR and SSIM over tf.float32 Tensors.
        im1 = new_image
        im2 = target_image

        mse = sess.run(tf.reduce_mean(tf.square(im1 - im2)))
        total_mse += mse

        psnr = sess.run(tf.image.psnr(im1, im2, max_val=1.0))
        total_psnr += psnr

        ssim = sess.run(tf.image.ssim(im1, im2, max_val=1.0))
        total_ssim += ssim

        message = 'Pic {} '.format(i) + 'mse={:5f}, '.format(
            mse) + 'psnr={:5f}, '.format(psnr) + 'ssmi={:5f}.'.format(ssim)

        print(message)

    average_mse = total_mse / count
    average_psnr = total_psnr / count
    average_ssim = total_ssim / count

    msg = 'average_mse={:5f}, '.format(
        average_mse) + 'average_psnr={:5f}, '.format(
            average_psnr) + 'average_ssim={:5f}.'.format(average_ssim)
    print(msg)


if __name__ == '__main__':
    main()
