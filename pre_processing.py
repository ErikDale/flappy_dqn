import tensorflow as tf


def pre_process(image):
    normalized_image = tf.image.per_image_standardization(image)
    gray_image = tf.image.rgb_to_grayscale(normalized_image)
    return gray_image