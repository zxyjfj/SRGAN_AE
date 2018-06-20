import os

import cv2
import numpy as np
import tensorflow as tf

from configs import *
from model import generator


def readpic(filename, shape=[1, 32, 32, 3], ext='jpg'):
    n, h, w, c = shape
    if ext == 'jpg' or ext == 'jpeg':
        decoder = tf.image.decode_jpeg
    elif ext == 'png':
        decoder = tf.image.decode_png

    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    img = decoder(value, channels=c)
    img = tf.image.crop_to_bounding_box(img, 0, 0, h, w)
    img = tf.to_float(img)

    t_image = tf.train.batch([img], batch_size=n, capacity=1)
    return t_image, key


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
    inferences = generator(test_image, is_GP=True)

    # ========================================
    #           Load Model
    # ========================================
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINTS_PATH)

    # ========================================
    #           Load Image
    # ========================================
    filename = os.path.join(TEST_DATA_PATH, 'small.png')
    img = readpic(filename)
    image = sess.run(img)

    sr_img = sess.run(inferences, feed_dict={test_image: image})

    new_img = np.reshape(sr_img, [sr_img[1], sr_img[2], sr_img[2]])

    cv2.imwrite(os.path.join(TEST_DATA_PATH, 'sr_img.png'), new_img)

    coord.request_stop()
    coord.join(threads=threads)


if __name__ == '__main__':
    main()
