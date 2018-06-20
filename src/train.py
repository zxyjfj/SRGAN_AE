import os
import time

import tensorflow as tf

import data_inputs
from configs import *
from model import code_discriminator, discriminator, encoder, generator
from utils import visualize_samples, load, save


def main():
    # 导入高分辨和低分辨的图片
    LR_batch, HR_batch = data_inputs.batch_queue_for_training(TRAIN_DATA_PATH)

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
    # ========================================
    #            Define Loss(WGAN/DCGAN有问题)
    # ========================================

    generator_loss = tf.reduce_mean(y_fake)
    discriminator_loss = tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)
    code_generator_loss = tf.reduce_mean(c_fake)
    code_discriminator_loss = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake)
    # generator_loss = tf.reduce_mean(-tf.log(y_fake + EPS))
    # discriminator_loss = tf.reduce_mean(
    #     -tf.log(y_real + EPS)) - tf.reduce_mean(-tf.log(1 - y_fake + EPS))
    # code_generator_loss = tf.reduce_mean(-tf.log(c_fake + EPS))
    # code_discriminator_loss = tf.reduce_mean(
    #     -tf.log(c_real + EPS)) - tf.reduce_mean(-tf.log(1 - c_fake + EPS))
    reconstruction_loss = tf.reduce_mean(
        tf.squared_difference(x_super_res, HR_holders))



    generator_encoder_loss = generator_loss + code_generator_loss \
    + reconstruction_loss_weight * reconstruction_loss

    # ----------------------------------------
    #               Generator

    generator_encoder_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        generator_encoder_loss, var_list=generator_vars + encoder_vars)

    # ----------------------------------------
    #               Discriminator

    discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        discriminator_loss, var_list=discriminator_vars)

    # ----------------------------------------
    #               Code Discriminator

    code_discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        code_discriminator_loss, var_list=code_discriminator_vars)

    # ========================================
    #               Important
    # ========================================
    d_clip = []
    for var in discriminator_vars:
        clip_bounds = [-0.01, 0.01]
        d_clip.append(
            tf.assign(var, tf.clip_by_value(var, clip_bounds[0],
                                            clip_bounds[1])))
    clip_disc_weight = tf.group(*d_clip)

    c_clip = []
    for var in code_discriminator_vars:
        clip_bounds = [-0.01, 0.01]
        c_clip.append(
            tf.assign(var, tf.clip_by_value(var, clip_bounds[0],
                                            clip_bounds[1])))
    clip_code_disc_weight = tf.group(*c_clip)

    # for summaries
    with tf.name_scope('Summary'):
        tf.summary.image('inputs', LR_holders, max_outputs=4)
        tf.summary.image('generator', x_super_res, max_outputs=4)
        tf.summary.image('targets', HR_holders, max_outputs=4)
        tf.summary.scalar('generator_loss', generator_loss)
        tf.summary.scalar('discriminator_loss', discriminator_loss)
        tf.summary.scalar('code_generator_loss', code_generator_loss)
        tf.summary.scalar('code_discriminator_loss', code_discriminator_loss)
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        tf.summary.scalar('generator_encoder_loss', generator_encoder_loss)

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
    # Merage all the summaries and write them out to TRAINING_DIR
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAIN_SUMMARY_PATH, sess.graph)

    num_item_per_epoch = len(os.listdir(TRAIN_DATA_PATH)) // BATCH_SIZE
    time_i = time.time()
    step = 0

    for epoch in range(NUM_EPOCH):
        for item in range(num_item_per_epoch):
            LR_images, HR_images = sess.run([LR_batch, HR_batch])

            feed_dict = {LR_holders: LR_images, HR_holders: HR_images}

            # ------------------train G twice-------------------
            _, gene_loss = sess.run(
                [generator_encoder_opt, generator_encoder_loss],
                feed_dict=feed_dict)
            _, gene_loss = sess.run(
                [generator_encoder_opt, generator_encoder_loss],
                feed_dict=feed_dict)
            # ------------------train D ------------------------
            _, d_loss = sess.run(
                [discriminator_opt, discriminator_loss], feed_dict=feed_dict)
            # ------------------train code_D -------------------
            _, c_loss = sess.run(
                [code_discriminator_opt, code_discriminator_loss],
                feed_dict=feed_dict)
            sess.run(
                [clip_disc_weight, clip_code_disc_weight], feed_dict=feed_dict)

            sr_img = sess.run(x_super_res, feed_dict=feed_dict)

            summary = sess.run(merged_summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=step)

            message = 'Epoch [{:3d}/{:3d}]'.format(epoch + 1, NUM_EPOCH) \
                + '[{:4d}/{:4d}]'.format(item + 1, num_item_per_epoch) \
                + 'gene_loss={:6.8f}, '.format(gene_loss) \
                + 'd_loss={:6.8f}, '.format(d_loss) \
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
