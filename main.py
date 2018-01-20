import numpy as np
import tensorflow as tf
import time
import tensorpack.dataflow.dataset as dataset
from datetime import datetime


def model(x1, x2):
    """
    A computational graph for CNN part.
    :param x1: full resolution image (320x260)
    :param x2: half resolution image
    :return: predicted heat map
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    n_filters1, n_filters2, n_filters3, n_filters4, n_filters5 = 128, 128, 128, 512, 256

    # TODO: n_joints handle properly
    # TODO: dimensions don't match. Do they use conv with 'SAME' sometimes?
    # TODO: local contrast normalization here.
    # TODO: create the second branch with x2
    # TODO: maybe 3x3 pooling with stride=2 is better

    x1 = conv_layer(x1, 5, 1, 3, n_filters1, 'conv1')  # result: 316x256
    x1 = max_pool_layer(x1, 2, 2)  # result: 158x128

    x1 = conv_layer(x1, 5, 1, n_filters1, n_filters2, 'conv2')  # result: 154x124
    x1 = max_pool_layer(x1, 3, 2)  # result: 77x64

    x1 = conv_layer(x1, 5, 1, n_filters2, n_filters3, 'conv3')  # result: 73x60

    x1 = conv_layer(x1, 9, 1, n_filters3, n_filters4, 'conv3')  # result: 65x52
    x1 = conv_layer(x1, 1, 1, n_filters4, n_filters5, 'conv3')  # result: 65x52
    heat_map_pred = conv_layer(x1, 1, 1, n_filters5, n_joints, 'conv3')  # result: 65x52

    return heat_map_pred


def mrf(heat_map):
    # TODO: produce a new heat map using MRF
    return heat_map


def conv2d(x, W, stride):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


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


def detection_rate(heat_map_pred, heat_map_target, normalized_radius=10):
    """
    For a given pixel radius normalized by the torso height of each sample, we count the number of images in
    the test set for which the distance between predicted position and ground truth is less than this radius.
    It was intoduced in [MODEC: Multimodal Decomposable Models for Human Pose Estimation]
    (https://homes.cs.washington.edu/~taskar/pubs/modec_cvpr13.pdf).

    heat_map_pred: tensor of size [n_images, height, width, n_joints]
    heat_map_target: tensor of size [n_images, height, width, n_joints]
    normalized_radius: pixel radius normalized by the torso height of each sample
    """
    # TODO: may be complicated, we need to know the torso height for each sample here (not only heat maps)
    pass


def read_flic():
    # TODO: read data
    # TODO: prepare 90x60 heat maps with gaussian blobs on the joints' locations
    pass


def get_dataset():
    # The simplest version is to import CIFAR-10 using [Tensorpack](https://github.com/ppwwyyxx/tensorpack)
    x_train, y_train, x_test, y_test = read_flic()

    # standardization
    x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
    x_train_pixel_std = x_train.std(axis=0)  # per-pixel std
    x_train = (x_train - x_train_pixel_mean) / x_train_pixel_std
    x_test = (x_test - x_train_pixel_mean) / x_train_pixel_std
    return x_train, y_train, x_test, y_test


time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
tb_folder = 'tb_ex10'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, cur_timestamp)
tb_train = '{}/{}/train'.format(tb_folder, cur_timestamp)
tb_test = '{}/{}/test'.format(tb_folder, cur_timestamp)
tb_log_iters = False

n_joints = 4  # TODO: how many?
x_train, y_train, x_test, y_test = get_dataset()
in_height, in_width, n_colors = x_train.shape[1:3]
hm_height, hm_width = y_train.shape[1:2]
n_train_subset = 50  # for debugging purposes

# Main hyperparameters
n_epochs = 10
batch_size = 1
lmbd = 0.0
lr = 0.05

with tf.device('/gpu:0'):
    x1 = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
    x2 = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_halfres')
    heat_map_target = tf.placeholder(tf.int64, [None, hm_height, hm_width, n_joints], name='heat_map')
    lr_tf = tf.placeholder(tf.float32)

    heat_map_pred = model(x1, x2)

    with tf.name_scope('loss'):
        mse = mean_squared_error(heat_map_pred, heat_map_target)
        loss = mse + lmbd * weight_decay(var_pattern='weights')

    with tf.name_scope('optimizer'):
        # opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
        grads_vars = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grads_vars)

    with tf.name_scope('evaluation'):
        det_rate_10 = detection_rate(heat_map_pred, heat_map_target, normalized_radius=10)

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

    tf.summary.image('input', x, 30)
    tf.summary.image('heat_map_pred', heat_map_pred, 30)

    tb_merged = tf.summary.merge_all()
    train_iters_writer = tf.summary.FileWriter(tb_train_iter)
    train_writer = tf.summary.FileWriter(tb_train)
    test_writer = tf.summary.FileWriter(tb_test)

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
    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))


# TODO: read dataset, prepare 90x60 target heat-maps (small gaussian with std=1.5 pixels wrt output space 90x60)

# TODO: set up a CNN: start from Fig.2.

# TODO: data augmentation during training as described in the paper:
# rotation [-20 .. 20], scale [0.5, 1.5], horiz. flip 0.5.

# TODO: set up evaluation

# TODO: save the model!!! and set up the option to continue training

# TODO: get first results and paste them into the report

# TODO: set up PGM part. How to implement large 128x128 convolutions efficiently? FFT in TF?
# Apply special initalization to the conv weights of PGM part, based on the histogram of joint displacements!

# TODO: CNN from Fig.4
# TODO: Data Augm.: try zero padding and random crops!
# TODO: spatial dropout


