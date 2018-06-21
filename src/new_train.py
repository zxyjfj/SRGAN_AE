# ============================================
#             Code For Mult-GPU
# ============================================
import os
import time

import tensorflow as tf

import data_inputs
from configs import *
from model import code_discriminator, discriminator, encoder, generator
from utils import visualize_samples, load, save


def main():
    # 初始化tensorflow
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # 有几块GPU
    N_GPUS = 2
    DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
    # 导入高分辨和低分辨的图片
    LR_batch, HR_batch = data_inputs.batch_queue_for_training(TRAIN_DATA_PATH)

    coord = tf.train.Coordinator()

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
    #           Create Network
    # ========================================
    LR_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    HR_holders = tf.placeholder(
        dtype=tf.float32,
        shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])

    Split_LR_holders = tf.split(LR_holders, len(DEVICES))
    Split_HR_holders = tf.split(HR_holders, len(DEVICES))

    gen_encoder_costs, disc_costs, code_disc_costs = [], [], []

    for device_index, (device, Split_LR_holders,
                       Split_HR_holders) in enumerate(
                           zip(DEVICES, LR_holders, HR_holders)):
        with tf.device(device):
            real_data = Split_HR_holders
            # ----------------------------------------
            #               Generator

            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                fake_data = generator(Split_LR_holders)

            # ----------------------------------------
            #               Discriminator

            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                disc_fake = discriminator(fake_data)

            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                disc_real = discriminator(real_data)

            # ----------------------------------------
            #               Encoder

            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                real_code = encoder(real_data)

            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                fake_code = encoder(fake_data)

            # ----------------------------------------
            #           Code Discriminator

            with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
                code_disc_fake = code_discriminator(fake_code)

            with tf.variable_scope('code_discriminator', reuse=tf.AUTO_REUSE):
                code_disc_real = code_discriminator(real_code)

            # ========================================
            #            Define Loss(WGAN/DCGAN有问题)
            # ========================================

            generator_loss = -tf.reduce_mean(disc_fake)
            discriminator_loss = tf.reduce_mean(disc_real) - tf.reduce_mean(
                disc_fake)
            code_generator_loss = -tf.reduce_mean(code_disc_fake)
            code_discriminator_loss = tf.reduce_mean(
                code_disc_real) - tf.reduce_mean(code_disc_fake)
            reconstruction_loss = tf.reduce_mean(
                tf.squared_difference(fake_data, real_data))



            generator_encoder_loss = generator_loss + code_generator_loss \
            + reconstruction_loss_weight * reconstruction_loss

            gen_encoder_costs.append(generator_encoder_loss)
            disc_costs.append(discriminator_loss)
            code_disc_costs.append(code_discriminator_loss)

    gen_encoder_cost = tf.add_n(gen_encoder_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)
    code_disc_cost = tf.add_n(code_disc_costs) / len(DEVICES)

    # ----------------------------------------
    #               Generator

    generator_encoder_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        gen_encoder_cost,
        var_list=generator_vars + encoder_vars,
        colocate_gradients_with_ops=True)

    # ----------------------------------------
    #               Discriminator

    discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        disc_cost,
        var_list=discriminator_vars,
        colocate_gradients_with_ops=True)

    # ----------------------------------------
    #               Code Discriminator

    code_discriminator_opt = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(
        code_disc_cost,
        var_list=code_discriminator_vars,
        colocate_gradients_with_ops=True)

    # ========================================
    #               Important
    # ========================================
    d_clip = []
    for var in discriminator_vars:
        clip_bounds = [-0.005, 0.005]
        d_clip.append(
            tf.assign(var, tf.clip_by_value(var, clip_bounds[0],
                                            clip_bounds[1])))
    clip_disc_weight = tf.group(*d_clip)

    c_clip = []
    for var in code_discriminator_vars:
        clip_bounds = [-0.005, 0.005]
        c_clip.append(
            tf.assign(var, tf.clip_by_value(var, clip_bounds[0],
                                            clip_bounds[1])))
    clip_code_disc_weight = tf.group(*c_clip)

    # for summaries
    with tf.name_scope('Summary'):
        tf.summary.image('inputs', LR_holders, max_outputs=4)
        tf.summary.image('generator', fake_data, max_outputs=4)
        tf.summary.image('targets', HR_holders, max_outputs=4)
        tf.summary.scalar('discriminator_loss', disc_cost)
        tf.summary.scalar('code_discriminator_loss', code_disc_cost)
        tf.summary.scalar('generator_encoder_loss', gen_encoder_cost)

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
    step = saved_global_step

    for epoch in range(NUM_EPOCH):
        for item in range(num_item_per_epoch):
            LR_images, HR_images = sess.run([LR_batch, HR_batch])

            feed_dict = {LR_holders: LR_images, HR_holders: HR_images}

            # ------------------train G twice-------------------
            _, gene_loss = sess.run(
                [generator_encoder_opt, gen_encoder_cost],
                feed_dict={
                    LR_holders: LR_images,
                    HR_holders: HR_images
                })
            for i in range(5):
                LR_image, HR_image = sess.run([LR_batch, HR_batch])

                feed_dict1 = {LR_holders: LR_image, HR_holders: HR_image}
                # ------------------train D ------------------------
                _, d_loss = sess.run(
                    [discriminator_opt, disc_cost], feed_dict=feed_dict1)
                # ------------------train code_D -------------------
                _, c_loss = sess.run(
                    [code_discriminator_opt, code_disc_cost],
                    feed_dict=feed_dict1)
                sess.run([clip_disc_weight, clip_code_disc_weight])

            sr_img = sess.run(fake_data, feed_dict=feed_dict)

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