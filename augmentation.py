import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_dataset():
    x_test = np.load('x_test_flic.npy')
    y_test = np.load('y_test_flic.npy')
    return x_test, y_test

def horizontal_flip(img1, img2):
    p = tf.random_uniform([1])[0]
    img1, img2 = tf.cond(tf.greater(p, 0.5),
                         lambda: (tf.image.flip_left_right(img1), tf.image.flip_left_right(img2)),
                         lambda: (img1, img2))
    return img1, img2

def random_rotation(img1, img2, max_rotate_angle):
    rand_angles = tf.random_uniform([1], minval=-max_rotate_angle, maxval=max_rotate_angle)[0]
    img1 = tf.contrib.image.rotate(img1, rand_angles)
    img2 = tf.contrib.image.rotate(img2, rand_angles)
    return img1, img2

# def crop(img1, img2):
#     h_rand = tf.random_uniform([1], minval=0, maxval=40)[0]
#     w_rand = tf.random_uniform([1], minval=0, maxval=40)[0]
#
#     img1 = tf.crop_and_resize(img1, [h_rand], crop_size=[in_height, in_width])
#     return img1, img2


def augment_each(img, hm):
    max_rotate_angle = np.pi / 9
    img, hm = horizontal_flip(img, hm)
    img, hm = random_rotation(img, hm, max_rotate_angle)
    # img, hm = crop(img, hm)
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
    return img, hm


def augment_train(img_tensor, hm_tensor):
    """
    data augmentation during training as described in the paper [Efficient Object Localization Using Convolutional
    Networks](https://arxiv.org/abs/1411.4280): rotation [-20 .. 20] degrees, scale [0.5, 1.5], horiz. flip 0.5.
    :param img_tensor:
    :return:
    """
    with tf.device('/cpu:0'):
        img_tensor, hm_tensor = tf.map_fn(lambda x: augment_each(x[0], x[1]), (img_tensor, hm_tensor))
        return img_tensor, hm_tensor

def reshape_img(img_ori, i):
    img = img_ori[i, :, :, :]
    print(img.shape)
    img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2]))
    return img

def reshape_hm(hm, i):
    hm = hm[i, :, :, 0]
    print(hm.shape)
    hm = np.reshape(hm, (hm.shape[0], hm.shape[1]))
    return hm


def show_img_augmentation(X, Y, sess):
    """This function is to visulize the result of data augmentation for testing"""
    i = 0  # show the i-th image
    img_ori, img_aug, hm_ori, hm_aug = sess.run([x_in, x1, y_in, hm_target], feed_dict={x_in: X, y_in: Y, flag_train: True})
    img_aug = reshape_img(img_aug, i)
    img_ori = reshape_img(img_ori, i)
    hm_ori = reshape_hm(hm_ori, i)
    hm_aug = reshape_hm(hm_aug, i)
    plt.figure(1)
    plt.imshow((img_ori))
    plt.figure(2)
    plt.imshow((img_aug))
    plt.figure(3)
    plt.imshow((hm_ori))
    plt.figure(4)
    plt.imshow((hm_aug))
    plt.show()

x_test, y_test = get_dataset()
x_test = x_test[0:10, :, :, :]
y_test = y_test[0:10, :, :, :]
n_joints = 9
n_train, in_height, in_width, n_colors = x_test.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
x_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints], name='heat_map')
flag_train = tf.placeholder(tf.bool, name='is_training')

x1, hm_target = tf.cond(flag_train, lambda: augment_train(x_in, y_in), lambda: (x_in, y_in))
x_batch, hm_target_batch = x1, hm_target


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('begin:')
    show_img_augmentation(x_test, y_test, sess)

