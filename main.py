import numpy as np
import os
import tensorflow as tf
import time

import augmentation
import data
import evaluation
from datetime import datetime

np.set_printoptions(suppress=True, precision=4)


def model(x1, x2, n_joints, debug=False):
    """
    A computational graph for CNN part.
    :param x1: full resolution image batch_size x 360 x 240 x 3
    :param x2: half resolution image batch_size x 180 x 120 x 3
    :return: predicted heat map batch_size x 90 x 60 x n_joints
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    n_filters1, n_filters2, n_filters3, n_filters4, n_filters5 = 128, 128, 128, 512, 256
    if debug:
        n_filters1, n_filters2, n_filters3, n_filters4, n_filters5 = n_filters1/8, n_filters2/8, n_filters3/8, n_filters4/8, n_filters5/8

    x1 = conv_layer(x1, 5, 1, 3, n_filters1, 'conv1_fullres')  # result: 360x240
    x1 = max_pool_layer(x1, 2, 2)  # result: 180x120
    x1 = conv_layer(x1, 5, 1, n_filters1, n_filters2, 'conv2_fullres')  # result: 180x120
    x1 = max_pool_layer(x1, 2, 2)  # result: 180x60
    x1 = conv_layer(x1, 5, 1, n_filters2, n_filters3, 'conv3_fullres')  # result: 90x60
    x1 = conv_layer(x1, 9, 1, n_filters3, n_filters4, 'conv4_fullres')  # result: 90x60

    x2 = conv_layer(x2, 5, 1, 3, n_filters1, 'conv1_halfres')  # result: 360x240
    x2 = max_pool_layer(x2, 2, 2)  # result: 180x120
    x2 = conv_layer(x2, 5, 1, n_filters1, n_filters2, 'conv2_halfres')  # result: 180x120
    x2 = max_pool_layer(x2, 2, 2)  # result: 180x60
    x2 = conv_layer(x2, 5, 1, n_filters2, n_filters3, 'conv3_halfres')  # result: 90x60
    x2 = conv_layer(x2, 9, 1, n_filters3, n_filters4, 'conv4_halfres')  # result: 90x60

    x = x1 + tf.image.resize_images(x2, [2*x2.shape[1], 2*x2.shape[2]])

    heat_map_pred = conv_layer(x, 9, 1, n_filters4, n_filters5, 'conv5')  # result: 90x60

    return heat_map_pred


def mrf(heat_map):
    # TODO: produce a new heat map using MRF
    return heat_map


def conv2d(x, W, stride):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def weight_variable(shape, fc=False):
    """weight_variable generates a weight variable of a given shape. Uses He initialization."""
    if not fc:
        n_in = shape[0] * shape[1] * shape[2]
        n_out = shape[0] * shape[1] * shape[3]
    else:
        n_in = shape[0]
        n_out = shape[1]
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / n_in))
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name='biases')


def conv_layer(x, size, stride, n_in, n_out, name):
    with tf.name_scope(name):
        w = weight_variable([size, size, n_in, n_out])
        b = bias_variable([n_out])
        pre_activ = conv2d(x, w, stride) + b
        activ = tf.nn.relu(pre_activ)
    # we do it out of the namescope to show it separately in Tensorboard
    tf.summary.image('f_activ_'+name, activ[:, :, :, 5:6], 20)
    return activ


def max_pool_layer(x, size, stride):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')


def fully_connected(x, n_in, n_out, name):
    with tf.name_scope(name):
        w = weight_variable([n_in, n_out], fc=True)
        b = bias_variable([n_out])
        return tf.matmul(x, w) + b


def get_next_batch(X, Y, batch_size):
    n_batches = len(X) // batch_size
    rand_idx = np.random.permutation(len(X))[:n_batches * batch_size]
    for batch_idx in rand_idx.reshape([n_batches, batch_size]):
        batch_x, batch_y = X[batch_idx], Y[batch_idx]
        yield batch_x, batch_y


def weight_decay(var_pattern):
    """
    L2 weight decay loss, based on all weights that have var_pattern in their name

    var_pattern - a substring of a name of weights variables that we want to use in Weight Decay.
    """
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(var_pattern) != -1:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)


def get_var_by_name(var_name_to_find):
    return [v for v in tf.trainable_variables() if v.name == var_name_to_find][0]


def mean_squared_error(hm1, hm2):
    """
    Mean squared error between 2 heat maps as described in
    [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

    hm1, hm2: tensor of size [n_images, height, width, n_joints]
    """
    return tf.reduce_sum((hm1 - hm2)**2, axis=[1, 2, 3]) / hm1.shape[3]  # we divide over number of joints


debug = True
model_path = 'models_ex'
time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
tb_folder = 'tb_ex10'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, cur_timestamp)
tb_train = '{}/{}/train'.format(tb_folder, cur_timestamp)
tb_test = '{}/{}/test'.format(tb_folder, cur_timestamp)
tb_log_iters = False

n_joints = 11
x_train, y_train, x_test, y_test = data.get_dataset('flic')
n_train, in_height, in_width, n_colors = x_train.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
if debug:
    n_train, n_test = 200, 50  # for debugging purposes we take only a small subset
    x_train, y_train, x_test, y_test = x_train[:n_train], y_train[:n_train], x_test[:n_test], y_test[:n_test]
# Main hyperparameters
n_epochs = 10
batch_size = 1
lmbd = 0.0
lr = 0.05
n_updates_total = n_epochs * n_train // batch_size
lr_decay_n_updates = [round(0.66*n_updates_total), round(0.8*n_updates_total), round(0.9*n_updates_total)]
lr_decay_coefs = [lr, lr/2, lr/5, lr/10]

with tf.device('/gpu:0'):
    x1_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
    hm_target_in = tf.placeholder(tf.int64, [None, hm_height, hm_width, n_joints], name='heat_map')
    flag_train = tf.placeholder(tf.bool, name='is_training')

    n_iters_tf = tf.Variable(0, trainable=False)

    lr_tf = tf.train.piecewise_constant(n_iters_tf, lr_decay_n_updates, lr_decay_coefs)

    # Data augmentation: we apply the same random transformations both to images and heat maps
    x1, hm_target = tf.cond(flag_train, lambda: augmentation.augment_train(x1_in, hm_target_in), lambda: (x1_in, hm_target_in))
    x2 = tf.image.resize_images(x1, [x1.shape[1]//2, x1.shape[2]//2])

    # The whole heat map prediction model is here
    heat_map_pred = model(x1, x2, n_joints, debug)
    heat_map_pred = mrf(heat_map_pred)

    with tf.name_scope('loss'):
        mse = mean_squared_error(heat_map_pred, heat_map_target)
        loss = mse + lmbd * weight_decay(var_pattern='weights')

    with tf.name_scope('optimizer'):
        # opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
        grads_vars = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grads_vars, global_step=n_iters_tf)

    with tf.name_scope('evaluation'):
        det_rate_10 = evaluation.detection_rate(heat_map_pred, heat_map_target, normalized_radius=10)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('error_rate', det_rate_10)
        # Add histograms for gradients (only for weights, not biases)
        for grad, var in grads_vars:
            if 'weights' in var.op.name:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                grad_l2_norm = tf.norm(tf.reshape(grad, [-1]))
                tf.summary.scalar(var.op.name + '/gradients', grad_l2_norm)
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    tf.summary.image('input', x1, 30)
    tf.summary.image('heat_map_pred', heat_map_pred, 30)

    tb_merged = tf.summary.merge_all()
    train_iters_writer = tf.summary.FileWriter(tb_train_iter)
    train_writer = tf.summary.FileWriter(tb_train)
    test_writer = tf.summary.FileWriter(tb_test)

    saver = tf.train.Saver()

gpu_options = tf.GPUOptions(visible_device_list='6', per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    summary = sess.run(tb_merged, feed_dict={x: x_train[:n_train_subset], y: y_train[:n_train_subset], lr_tf: lr})
    train_writer.add_summary(summary, 0)
    summary = sess.run(tb_merged, feed_dict={x: x_test[:n_train_subset], y: y_test[:n_train_subset], lr_tf: lr})
    test_writer.add_summary(summary, 0)

    global_iter = 0
    for epoch in range(1, n_epochs + 1):
        # lr = lr/5 if epoch == int(n_epochs*0.7) else lr
        # lr = lr/5 if epoch == int(n_epochs*0.85) else lr
        # lr = lr/5 if epoch == int(n_epochs*0.95) else lr
        for x_train_batch, y_train_batch in get_next_batch(x_train, y_train, batch_size):
            global_iter += 1
            if tb_log_iters:
                _, summary = sess.run([train_step, tb_merged], feed_dict={x: x_train_batch, y: y_train_batch, lr_tf: lr})
                train_iters_writer.add_summary(summary, global_iter)
            else:
                sess.run(train_step, feed_dict={x: x_train_batch, y: y_train_batch, lr_tf: lr})

        summary = sess.run(tb_merged, feed_dict={x: x_train[:n_train_subset], y: y_train[:n_train_subset], lr_tf: lr})
        train_writer.add_summary(summary, epoch)
        summary = sess.run(tb_merged, feed_dict={x: x_test[:n_train_subset], y: y_test[:n_train_subset], lr_tf: lr})
        test_writer.add_summary(summary, epoch)

        test_err = eval_error(x_test, y_test, sess, batch_size)
        # Note, we evaluate the training error only on 1000 examples batch due to limited computational power
        train_err = eval_error(x_train[:n_train_subset], y_train[:n_train_subset], sess, batch_size)
        print('Epoch: {:d}  test err: {:.3f}%  train err: {:.3f}%'.format(epoch, test_err * 100, train_err * 100))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    saver.save(sess, model_path + '/' + cur_timestamp)  # save TF model for future real robustness test

    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))

# TODO: preprocess images /255, preprocess heat maps / 256.

# TODO: Eval, TB, batches, clean up

# TODO: debug

# TODO: get first results with part-detector only and paste them into the report

# TODO: set up PGM part.
# TODO: Apply special initalization to the conv weights of PGM part, based on the histogram of joint displacements!


# Things that are not super important
# TODO: data: handle multiple people by incorporating an extra "torso-joint heatmap" (page 6)
# TODO: set up the option to continue training (since we should do it in 3 stages according to the paper)
# TODO: local contrast normalization: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf and
# https://github.com/bowenbaker/metaqnn/blob/master/libs/input_modules/preprocessing.py
# TODO: Data Augm.: try zero padding and random crops!
# TODO: maybe 3x3 pooling with stride=2 is better
# TODO: maybe add 1x1 conv layer in the end?
# TODO: spatial dropout
# TODO: advanced pgm on small video clips from movies: http://bensapp.github.io/videopose-dataset.html

