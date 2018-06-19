import os
import time

import tensorflow as tf

import data_inputs
from configs import *
from model import code_discriminator, discriminator, encoder, generator
from utils import visualize_samples, load, save


def main():
    # 导入高分辨和低分辨的图片
    LR_batch, HR_batch = data_inputs.batch_queue_for_training(
        TRAINING_DATA_PATH)

    coord = tf.train.Coordinator()

    # ========================================
    #           Create Network
    # ========================================
    LR_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    HR_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])

    # ----------------------------------------
    #               Generator

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x_super_res = generator(LR_holders)

    # ----------------------------------------
    #               Discriminator

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        y_fake = discriminator(x_super_res)

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        y_real = discriminator(HR_holders)

    # ----------------------------------------
    #               Encoder

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        HR_encoded = encoder(HR_holders)

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x_encoded = encoder(x_super_res)

    # ----------------------------------------
    #           Code Discriminator

    with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
        c_fake = code_discriminator(x_encoded)

    with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
        c_real = code_discriminator(HR_encoded)

    # ========================================
    #            Define Loss
    # ========================================

    generator_loss = tf.reduce_mean(y_fake)
    discriminator_loss = tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)
    code_generator_loss = tf.reduce_mean(c_fake)
    code_discriminator_loss = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake)
    mse_loss = tf.reduce_mean(tf.squared_difference(x_super_res, HR_holders))

    # ========================================
    #            Create Optimizer
    # ========================================

    variables = tf.trainable_variables()
    encoder_vars = [var for var in variables if 'encoder/' in var.name]
    generator_vars = [var for var in variables if 'generator/' in var.name]
    discriminator_vars = [
        var for var in variables if 'discriminator/' in var.name
    ]
    code_discriminator_vars = [
        var for var in variables if 'code_discriminator/' in var.name
    ]

    # ----------------------------------------
    #               Encoder
    encoder_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        code_generator_loss, var_list=encoder_vars)

    # ----------------------------------------
    #               Generator

    generator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        generator_loss, var_list=generator_vars)

    # ----------------------------------------
    #               Discriminator

    discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        discriminator_loss, var_list=discriminator_vars)

    # ----------------------------------------
    #               Code Discriminator

    code_discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        code_discriminator_loss, var_list=code_discriminator_vars)

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

    # Start the queue runners (make batches).
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_item_per_epoch = len(os.listdir(TRAINING_DATA_PATH)) // BATCH_SIZE
    time_i = time.time()
    step = 0

    for epoch in range(NUM_EPOCH):
        for item in range(num_item_per_epoch):
            LR_images, HR_images = sess.run([LR_batch, HR_batch])

            feed_dict = {LR_holders: LR_images, HR_holders: HR_images}

            # ------------------train G twice-------------------
            _, g_loss = sess.run(
                [generator_opt, generator_loss], feed_dict=feed_dict)
            _, g_loss = sess.run(
                [generator_opt, generator_loss], feed_dict=feed_dict)
            # ------------------train D ------------------------
            _, d_loss = sess.run(
                [discriminator_opt, discriminator_loss], feed_dict=feed_dict)
            # ------------------train encoder ------------------
            _, e_loss = sess.run(
                [encoder_opt, code_discriminator_loss], feed_dict=feed_dict)
            # ------------------train code_D -------------------
            _, c_loss = sess.run(
                [code_discriminator_opt, code_discriminator_loss],
                feed_dict=feed_dict)

            sr_img = sess.run(x_super_res, feed_dict=feed_dict)

            message = 'Epoch [{:3d}/{:3d}]'.format(epoch + 1, NUM_EPOCH) \
                + '[{:4d}/{:4d}]'.format(item + 1, num_item_per_epoch) \
                + 'g_loss={:6.8f}, '.format(g_loss) \
                + 'd_loss={:6.8f}, '.format(d_loss) \
                + 'e_loss={:6.8f}, '.format(e_loss) \
                + 'c_loss={:6.8f}, '.format(c_loss) \
                + 'Time={:.2f}.'.format(time.time() - time_i)
            print(message)
            step += 1

        visualize_samples(
            sess,
            HR_images,
            sr_img,
            filename=os.path.join(INFERENCES_SAVE_PATH,
                                  'trian-epoch-{:03d}.png'.format(epoch + 1)))

        save(saver, sess, CHECKPOINTS_PATH, step)

    coord.request_stop()
    coord.join(threads=threads)


if __name__ == '__main__':
    main()
