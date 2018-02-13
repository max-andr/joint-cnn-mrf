import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_dataset():
    x_test = np.load('x_test_flic.npy')
    y_test = np.load('y_test_flic.npy')
    return x_test, y_test


def random_flip(img, hm):
    """
    Note, here we have to handle carefully left and right joints after reflection
    :param img: is the image itself
    :param hm: is the heat map, 60 x 90 x 10
    """
    img = tf.image.flip_left_right(img)
    hm = tf.image.flip_left_right(hm)
    hm = tf.concat([hm[:, :, 3:6], hm[:, :, 0:3], hm[:, :, 7:8], hm[:, :, 6:7], hm[:, :, 8:10]], axis=2)
    return img, hm


def horizontal_flip(img, hm):
    p = tf.random_uniform([1])[0]
    img, hm = tf.cond(tf.greater(p, 0.5),
                      lambda: random_flip(img, hm),
                      lambda: (img, hm))
    return img, hm


def random_rotation(img, hm, max_rotate_angle):
    rand_angles = tf.random_uniform([1], minval=-max_rotate_angle, maxval=max_rotate_angle)[0]
    img = tf.contrib.image.rotate(img, rand_angles, interpolation='BILINEAR')
    hm = tf.contrib.image.rotate(hm, rand_angles, interpolation='BILINEAR')
    return img, hm


def random_crop(img, hm):
    crop_size = 0.95  # relative size of the cropped region
    img = tf.expand_dims(img, 0)
    hm = tf.expand_dims(hm, 0)
    h1, w1 = img.shape[1].value, img.shape[2].value
    h2, w2 = hm.shape[1].value, hm.shape[2].value
    rh = tf.random_uniform([1], minval=0, maxval=1-crop_size)[0]
    rw = tf.random_uniform([1], minval=0, maxval=1-crop_size)[0]

    img = tf.image.crop_and_resize(img, boxes=[[rh, rw, rh + crop_size, rw + crop_size]], crop_size=[h1, w1], box_ind=[0])
    hm = tf.image.crop_and_resize(hm, boxes=[[rh, rw, rh + crop_size, rw + crop_size]], crop_size=[h2, w2], box_ind=[0])
    # goal: keep roughly 3x3 gaussian kernel in the joint location (after crop+resize we have often 4x4 or 5x5
    # which perform suboptimally)
    hm = hm**1.6  # kind of non-maxima suppression
    hm += 10 ** -5  # to prevent division by 0; also can be seen as label smoothing
    hm = hm / tf.reduce_sum(hm, axis=[1, 2], keep_dims=True) # make sure that the resized HM is a probability distr
    return img[0], hm[0]  # get rid of the 1st fake dimension


def augment_train(img_tensor, hm_tensor):
    """
    data augmentation during training as described in the paper [Efficient Object Localization Using Convolutional
    Networks](https://arxiv.org/abs/1411.4280): rotation [-20 .. 20] degrees, scale [0.5, 1.5], horiz. flip 0.5.
    :param img_tensor:
    :return:
    """
    def augment_each_train(img, hm):
        max_rotate_angle = np.pi / 9
        img, hm = horizontal_flip(img, hm)
        img = tf.image.random_brightness(img, max_delta=32. / 255.)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
        img, hm = random_rotation(img, hm, max_rotate_angle)
        img, hm = random_crop(img, hm)
        # img = tf.image.per_image_standardization(img)
        return img, hm

    img_tensor, hm_tensor = tf.map_fn(lambda x: augment_each_train(x[0], x[1]), (img_tensor, hm_tensor))
    return img_tensor, hm_tensor


def augment_test(img_tensor, hm_tensor):
    """
    data augmentation during testing
    """
    def augment_each_train(img, hm):
        # img = tf.image.per_image_standardization(img)
        return img, hm

    img_tensor, hm_tensor = tf.map_fn(lambda x: augment_each_train(x[0], x[1]), (img_tensor, hm_tensor))
    return img_tensor, hm_tensor


def reshape_img(img_ori, i):
    img = img_ori[i, :, :, :]
    print(img.shape)
    return img


def reshape_hm(hm, i):
    hm = hm[i, :, :, 8]  # show nose
    print(hm.shape)
    return hm


def show_img_augmentation(X, Y, sess):
    """This function is to visulize the result of data augmentation for testing"""
    i = 0  # show the i-th image
    img_ori, img_aug, hm_ori, hm_aug = sess.run([x_in, x_batch, y_in, hm_target_batch], feed_dict={x_in: X, y_in: Y})
    img_aug = reshape_img(img_aug, i)
    img_ori = reshape_img(img_ori, i)
    hm_ori = reshape_hm(hm_ori, i)
    hm_aug = reshape_hm(hm_aug, i)
    # plt.figure(1)
    # plt.imshow(img_ori)
    # plt.savefig('img/img_orig.png', dpi=300)
    # plt.clf()

    plt.figure(2)
    plt.imshow(img_aug)
    plt.savefig('img/img_augm.png', dpi=300)
    plt.clf()

    # plt.figure(3)
    # plt.imshow(hm_ori)
    # plt.savefig('img/hm_orig.png', dpi=300)
    # plt.clf()

    plt.figure(4)
    plt.imshow(hm_aug)
    plt.savefig('img/hm_augm.png', dpi=300)
    plt.clf()


if __name__ == '__main__':
    x_test, y_test = get_dataset()
    x_test = x_test[:10]
    y_test = y_test[:10]
    n_joints = 10
    n_train, in_height, in_width, n_colors = x_test.shape[0:4]
    n_test, hm_height, hm_width = y_test.shape[0:3]

    with tf.device('/cpu:0'):
        x_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
        y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints], name='heat_map')

        x_batch, hm_target_batch = augment_train(x_in, y_in)

    gpu_options = tf.GPUOptions(visible_device_list='7', per_process_gpu_memory_fraction=0.05)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('begin:')
        show_img_augmentation(x_test, y_test, sess)

