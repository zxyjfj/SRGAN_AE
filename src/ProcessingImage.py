import os
from glob import glob

import tensorflow as tf

from configs import INPUT_SIZE, NUM_CHENNELS, PATCH_SIZE, TEST_DATA_PATH, TEST_GROUND_TRUTH


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

    save_filename1 = TEST_DATA_PATH + '/%s.png' % (
        os.path.basename(filename).split('.')[0])

    save_filename2 = TEST_GROUND_TRUTH + '/%s.png' % (
        os.path.basename(filename).split('.')[0])

    with tf.gfile.FastGFile(save_filename1, 'wb') as f:
        lr_image = tf.image.convert_image_dtype(lr_image, tf.uint8)
        f.write(sess.run(tf.image.encode_png(lr_image)))

    with tf.gfile.FastGFile(save_filename2, 'wb') as f:
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        f.write(sess.run(tf.image.encode_png(image)))


if __name__ == '__main__':
    with tf.Session() as sess:
        init = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ]
        sess.run(init)
        file_list = glob('../data/imgs/*.jpg')
        for filename in file_list:
            ProcessingImage(sess, filename)
