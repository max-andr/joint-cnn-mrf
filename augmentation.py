import tensorflow as tf
import numpy as np


def horizontal_flip(img1, img2):
    p = tf.random_uniform([1])[0]
    img1, img2 = tf.cond(tf.greater(p, 0.5),
                         lambda: tf.image.random_flip_left_right(img1), tf.image.random_flip_left_right(img2),
                         (img1, img2))
    return img1, img2


# def crop(img1, img2):
#     h_rand = tf.random_uniform([1], minval=0, maxval=40)[0]
#     w_rand = tf.random_uniform([1], minval=0, maxval=40)[0]
#
#     img1 = tf.crop_and_resize(img1, [h_rand], crop_size=[in_height, in_width])
#     return img1, img2


def augment_each(img, hm):
    img, hm = horizontal_flip(img, hm)
    # img, hm = crop(img, hm)
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    return img


def augment_train(img_tensor, hm_tensor):
    """
    data augmentation during training as described in the paper [Efficient Object Localization Using Convolutional
    Networks](https://arxiv.org/abs/1411.4280): rotation [-20 .. 20] degrees, scale [0.5, 1.5], horiz. flip 0.5.
    :param img_tensor:
    :return:
    """
    # TODO: finish random cropping
    max_rotate_angle = np.pi/9
    batch_size = img_tensor.shape[0]
    with tf.device('/cpu:0'):
        img_tensor, hm_tensor = tf.map_fn(augment_each, (img_tensor, hm_tensor))

        rand_angles = tf.random_uniform([batch_size], minval=-max_rotate_angle, maxval=max_rotate_angle)
        img_tensor = tf.contrib.image.rotate(img_tensor, rand_angles)
        hm_tensor = tf.contrib.image.rotate(hm_tensor, rand_angles)
        return img_tensor, hm_tensor
