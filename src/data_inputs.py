from os.path import join

import tensorflow as tf
from configs import *


def batch_queue_for_training(data_path):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(join(data_path, '*.png')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHENNELS)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    # if MAX_RANDOM_BRIGHTNESS > 0:
    #     patch = tf.image.random_brightness(patch, MAX_RANDOM_BRIGHTNESS)
    # if len(RANDOM_CONTRAST_RANGE) == 2:
    #     patch = tf.image.random_contrast(patch, *RANDOM_CONTRAST_RANGE)
    high_res_patch = patch

    crop_margin = PATCH_SIZE - LABEL_SIZE
    assert crop_margin >= 0
    if crop_margin > 1:
        high_res_patch = tf.random_crop(patch,
                                        [LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
        # crop_pos = tf.random_uniform([2], 0, crop_margin, dtype=tf.int32)
        # offset = tf.convert_to_tensor([crop_pos[0], crop_pos[1], 0])
        # size = tf.convert_to_tensor([CROP_SIZE, CROP_SIZE, NUM_CHENNELS])
        # high_res_patch = tf.slice(patch, offset, size)
        # additional 1px shifting to low_res_patch, reducing the even/odd issue in nearest neighbor scaler.
        # shift1px = tf.random_uniform([2], -1, 2, dtype=tf.int32)
        # offset += tf.convert_to_tensor([shift1px[0], shift1px[1], 0])
        # offset = tf.clip_by_value(offset, 0, crop_margin-1)
        # low_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE, INPUT_SIZE]

    def resize_nn():
        return tf.image.resize_nearest_neighbor([high_res_patch],
                                                downscale_size, True)

    def resize_area():
        return tf.image.resize_area([high_res_patch], downscale_size, True)

    def resize_cubic():
        return tf.image.resize_bicubic([high_res_patch], downscale_size, True)

    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case(
        {
            tf.equal(r, 0): resize_nn,
            tf.equal(r, 1): resize_area
        },
        default=resize_cubic)[0]

    # add jpeg noise to low_res_patch
    # if JPEG_NOISE_LEVEL > 0:
    #     low_res_patch = tf.image.convert_image_dtype(
    #         low_res_patch, dtype=tf.uint8, saturate=True)
    #     jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
    #     jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
    #     low_res_patch = tf.image.decode_jpeg(jpeg_code)
    #     low_res_patch = tf.image.convert_image_dtype(
    #         low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    # if GAUSSIAN_NOISE_STD > 0:
    #     low_res_patch += tf.random_normal(
    #         low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # low_res_patch -= 0.5    # approximate mean-zero data
    # high_res_patch -= 0.5
    # low_res_patch = tf.clip_by_value(low_res_patch, -0.5, 0.5)
    # high_res_patch = tf.clip_by_value(high_res_patch, -0.5, 0.5)

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
        tf.train.match_filenames_once(join(data_path, '*.jpg')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_jpeg(image_file, NUM_CHENNELS)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    offset = tf.convert_to_tensor([crop_margin // 2, crop_margin // 2, 0])
    size = tf.convert_to_tensor([LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
    high_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE, INPUT_SIZE]

    def resize_nn():
        return tf.image.resize_nearest_neighbor([high_res_patch],
                                                downscale_size, True)

    def resize_area():
        return tf.image.resize_area([high_res_patch], downscale_size, True)

    def resize_cubic():
        return tf.image.resize_bicubic([high_res_patch], downscale_size, True)

    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case(
        {
            tf.equal(r, 0): resize_nn,
            tf.equal(r, 1): resize_area
        },
        default=resize_cubic)[0]

    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(
            low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # low_res_patch -= 0.5  # approximate mean-zero data
    # high_res_patch -= 0.5
    # low_res_patch = tf.clip_by_value(low_res_patch, -0.5, 0.5)
    # high_res_patch = tf.clip_by_value(high_res_patch, -0.5, 0.5)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE)

    return low_res_batch, high_res_batch
