import pandas as pd
import tensorflow as tf
import numpy as np
import config

def load_list_from_df(file_path):
    _df = pd.read_csv(file_path)
    input_data = _df['input'].tolist()
    groud_truth_data = _df['ground_truth'].tolist()
    return input_data, groud_truth_data

def get_labels(file_path):
    file_path_truncated = tf.strings.split(file_path, sep='.')
    labeled_file_path = tf.strings.join([file_path_truncated[0] + '_label.png'])
    img = tf.io.read_file(labeled_file_path)
    return tf.image.decode_png(img, channels=1)

def get_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def one_hot_encode(label):
    flatten_label = tf.reshape(label, [config.WIDTH * config.HEIGHT])
    oh = tf.one_hot(flatten_label, depth=19)
    oh_reshaped = tf.reshape(oh, [config.HEIGHT, config.WIDTH, 19])
    return oh_reshaped

def random_crop(image, label):
    label = tf.cast(label, dtype=tf.float32)
    stacked = tf.concat([image, label], axis=-1)
    cropped_stacked = tf.image.random_crop(stacked, [416, 832, 4])
    image, label = tf.split(cropped_stacked, [3, 1], axis=-1)
    label = tf.cast(label, dtype=tf.uint8)
    return image, label

def random_left_right_flip(image, label):
    label = tf.cast(label, dtype=tf.float32)
    stacked = tf.concat([image, label], axis=-1)
    cropped_stacked = tf.image.flip_left_right(stacked)
    image, label = tf.split(cropped_stacked, [3, 1], axis=-1)
    label = tf.cast(label, dtype=tf.uint8)
    return image, label

def training_image_label_pairs_processing(file_path):
    label = get_labels(file_path)
    image = get_image(file_path)
    image, label = random_crop(image, label)
    image = tf.image.resize(image, [config.HEIGHT, config.WIDTH], method='bilinear')
    label = tf.image.resize(label, [config.HEIGHT, config.WIDTH], method='nearest')
    image, label = random_left_right_flip(image, label)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_hue(image, 0.08)
    image = (image - config.MEANS) / config.STD
    label = tf.image.convert_image_dtype(label, dtype=tf.uint8)
    image.set_shape(shape=(config.HEIGHT, config.WIDTH, 3))
    label.set_shape(shape=(config.HEIGHT, config.WIDTH, 1))
    label = one_hot_encode(label)
    return image, label

def val_image_label_pairs_processing(file_path):
    label = get_labels(file_path)
    image = get_image(file_path)
    label = tf.image.resize(label, (config.HEIGHT, config.WIDTH), method='nearest')
    image = tf.image.resize(image, (config.HEIGHT, config.WIDTH), method='bilinear')
    image = (image - config.MEANS) / config.STD
    label = tf.image.convert_image_dtype(label, dtype=tf.uint8)
    image.set_shape(shape=(config.HEIGHT, config.WIDTH, 3))
    label.set_shape(shape=(config.HEIGHT, config.WIDTH, 1))
    label = one_hot_encode(label)
    return image, label

def preparation_for_training(dataset, batch_size, shuffle_buffer_size=32):
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=config.AUTOTUNE)
    return dataset