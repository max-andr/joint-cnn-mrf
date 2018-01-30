import numpy as np
import os
import tensorflow as tf
import time
import pickle

import augmentation
import evaluation
import tensorboard as tb
from datetime import datetime

joint_names = np.array(['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose'])

joint_dependence = {'lsho': ['nose', 'lelb'], 'lelb': ['lsho', 'lwri'], 'lwri': ['lelb'],
                    'rsho': ['nose', 'relb'], 'relb': ['rsho', 'rwri'], 'rwri': ['relb'],
                    'lhip': ['nose'], 'rhip': ['nose'], 'nose': ['lsho', 'rsho', 'lhip', 'rhip']}

dict = {'lsho':   0, 'lelb': 1, 'lwri': 2, 'rsho': 3, 'relb': 4, 'rwri': 5, 'lhip': 6,
        'lkne':   7, 'lank': 8, 'rhip': 9, 'rkne': 10, 'rank': 11, 'leye': 12, 'reye': 13,
        'lear':   14, 'rear': 15, 'nose': 16, 'msho': 17, 'mhip': 18, 'mear': 19, 'mtorso': 20,
        'mluarm': 21, 'mruarm': 22, 'mllarm': 23, 'mrlarm': 24, 'mluleg': 25, 'mruleg': 26,
        'mllleg': 27, 'mrlleg': 28}

np.set_printoptions(suppress=True, precision=4)


def model(x, n_joints, debug=False):
    """
    A computational graph for CNN part.
    :param x1: full resolution image batch_size x 360 x 240 x 3
    :param x2: half resolution image batch_size x 180 x 120 x 3
    :return: predicted heat map batch_size x 90 x 60 x n_joints
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    n_bottleneck, n_filters1, n_filters2, n_filters3, n_filters4, n_filters5 = 64, 64, 128, 256, 256, 512
    # if debug:
    #     n_filters1, n_filters2, n_filters3, n_filters4, n_filters5 = \
    #         n_filters1 // 8, n_filters2 // 8, n_filters3 // 8, n_filters4 // 8, n_filters5 // 8

    x1 = x
    x2 = tf.image.resize_images(x, [int(x.shape[1]) // 2, int(x.shape[2]) // 2])

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

    x = x1 + tf.image.resize_images(x2, [2 * int(x2.shape[1]), 2 * int(x2.shape[2])])
    x = conv_layer(x, 9, 1, n_filters4, n_filters5, 'conv5')  # result: 90x60
    x = conv_layer(x, 9, 1, n_filters5, n_joints, 'conv6', last_layer=True)  # result: 90x60

    return x


def get_pairwise_distribution(joint, cond, pairwise_distribution):
    """
    get a specific frame from pairwise_distribution, which is 180 x 120 x n_pairs
    :param joint: the joint name: string
    :param cond: the name of the joint that is conditioned on: string 
    :param pairwise_distribution: 180 x 120 x n_pairs, it's dictionary with the key as 'lsho_nose' meaning P of lsho given nose
    :return: 1x180 x 120 x 1
    """
    return pairwise_distribution[joint + '_' + cond]


def conv_mrf(A, B):
    """
    
    :param A: conv kernel 1 x 120 x 180 x1
    :param B: input heatmaps: batch_size x 60 x 90 x 1
    :return: C is batch_size x 60 x 90 x 1
    """
    # B = tf.map_fn(lambda img: tf.image.flip_left_right(tf.image.flip_up_down(img)), B)
    B = tf.reshape(B, [hm_height, hm_width, 1, tf.shape(B)[0]])
    B = tf.reverse(B, axis=[0, 1])

    # conv between 1 x 120 x 180 x 1 and 60 x 90 x 1 x ? => 1 x 61 x 91 x ?
    C = tf.nn.conv2d(A, B, strides=[1, 1, 1, 1], padding='VALID')  # 1 x 91 x 61 x 1
    C = C[:, :hm_height, :hm_width, :]
    C = tf.reshape(C, [tf.shape(B)[3], hm_height, hm_width, 1])
    # return tf.image.crop_to_bounding_box(C, 0, 0, 60, 90)
    return C


def mrf_fixed(heat_map, pairwise_distribution):
    """
    from Learning Human Pose Estimation Features with Convolutional Networks
    :param heat_map: is produced by model as the unary distributions: batch_size x 90 x 60 x n_joints
    :param pairwise_distribution: the priors for a pair of joints, it's calculated from the histogram and is fixed 1x180x120xn_pairs
    :return heat_map_hat: the result from equation 1 in the paper: batch_size x 90 x 60 x n_joints
    """
    delta = 10**-7
    heat_map_hat = []
    for joint_id, joint_name in enumerate(joint_names):
        # log_p_joint = tf.log(heat_map[:, :, :, joint_id:joint_id+1] + delta)  # heat_map: batch_size x 90 x 60 x 1
        p_joint = heat_map[:, :, :, joint_id:joint_id+1]  # heat_map: batch_size x 90 x 60 x 1
        for cond_joint in joint_dependence[joint_name]:
            cond_joint_id = np.where(joint_names == cond_joint)[0][0]
            p_joint *= conv_mrf(get_pairwise_distribution(joint_name, cond_joint, pairwise_distribution),
                                heat_map[:, :, :, cond_joint_id:cond_joint_id+1])
        heat_map_hat.append(p_joint)
    return tf.stack(heat_map_hat, axis=3)[:, :, :, :, 0]


def mrf_trainable(heat_map):
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


def conv_layer(x, size, stride, n_in, n_out, name, last_layer=False):
    with tf.name_scope(name):
        w = weight_variable([size, size, n_in, n_out])
        b = bias_variable([n_out])
        pre_activ = conv2d(x, w, stride) + b
        if not last_layer:
            activ = tf.nn.relu(pre_activ)
            activ = tf.contrib.layers.batch_norm(activ, decay=0.9, center=True, scale=True, is_training=flag_train)
        else:
            activ = pre_activ
    # we do it out of the namescope to show it separately in Tensorboard
    tb.var_summary(pre_activ, name)
    tf.summary.image('f_activ_' + name, activ[:, :, :, 7:8], 20)
    return activ


def max_pool_layer(x, size, stride):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')


def fully_connected(x, n_in, n_out, name):
    with tf.name_scope(name):
        w = weight_variable([n_in, n_out], fc=True)
        b = bias_variable([n_out])
        return tf.matmul(x, w) + b


def get_next_batch(X, Y, batch_size, shuffle=False):
    n_batches = len(X) // batch_size
    if shuffle:
        x_idx = np.random.permutation(len(X))[:n_batches * batch_size]
    else:
        x_idx = np.arange(len(X))
    for batch_idx in x_idx.reshape([n_batches, batch_size]):
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
    Mean squared error between 2 heat maps for a batch as described in
    [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

    hm1, hm2: tensor of size [n_images, height, width, n_joints]
    """
    # if we don't multiply by number of pixels, then we get too small value of the loss
    return tf.reduce_mean((hm1 - hm2) ** 2) * hm_height * hm_width * n_joints


def eval_error(X_np, Y_np, sess, batch_size):
    """Get all predictions for a dataset by running it in small batches."""
    n_batches = len(X_np) // batch_size
    mse_pd_val, mse_sm_val = 0.0, 0.0
    for batch_x, batch_y in get_next_batch(X_np, Y_np, batch_size):
        val1, val2 = sess.run([mse_pd, mse_sm], feed_dict={x_in: batch_x, y_in: batch_y, flag_train: False})
        mse_pd_val, mse_sm_val = mse_pd_val + val1, mse_sm_val + val2
    return mse_pd_val / n_batches, mse_sm_val / n_batches


def spatial_softmax(hm):
    hm_height, hm_width, n_joints = int(hm.shape[1]), int(hm.shape[2]), int(hm.shape[3])
    hm = tf.reshape(hm, [-1, hm_height * hm_width, n_joints])
    hm = tf.nn.softmax(logits=hm, dim=1)
    hm = tf.reshape(hm, [-1, hm_height, hm_width, n_joints])
    return hm


def get_dataset():
    x_train = np.load('x_train_flic.npy')
    x_test = np.load('x_test_flic.npy')
    y_train = np.load('y_train_flic.npy')
    y_test = np.load('y_test_flic.npy')
    return x_train, y_train, x_test, y_test


gpu_number, gpu_memory = '6', 0.6
debug = True
train, restore_model, best_model_name = False, True, '2018-01-29 15:24:39'
model_path = 'models_ex'
time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
tb_folder = 'tb'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, cur_timestamp)
tb_train = '{}/{}/train'.format(tb_folder, cur_timestamp)
tb_test = '{}/{}/test'.format(tb_folder, cur_timestamp)
tb_log_iters = False
# img_tb_from, img_tb_to = 450, 465
img_tb_from, img_tb_to = 50, 65
n_eval_ex = 500

n_joints = 9
n_pairs = 8
x_train, y_train, x_test, y_test = get_dataset()
n_train, in_height, in_width, n_colors = x_train.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
if debug:
    n_train, n_test = 70, 70  # for debugging purposes we take only a small subset
    x_train, y_train, x_test, y_test = x_train[:n_train], y_train[:n_train], x_test[:n_test], y_test[:n_test]
# Main hyperparameters
n_epochs = 15
batch_size = 10
lmbd = 0.00000
lr, optimizer = 0.001, 'adam'  # So far best without BN: 0.001, 'adam'
n_updates_total = n_epochs * n_train // batch_size
lr_decay_n_updates = [round(0.7 * n_updates_total), round(0.8 * n_updates_total), round(0.9 * n_updates_total)]
lr_decay_coefs = [lr, lr / 2, lr / 5, lr / 10]

with open('pairwise_distribution.pickle', 'rb') as handle:
    pairwise_distr_np = pickle.load(handle)

with tf.device('/gpu:0'):
    x_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
    pairwise_distribution = {}
    for joint in joint_names:
        for cond_joint in joint_dependence[joint]:
            joint_key = joint+'_'+cond_joint
            tensor = tf.convert_to_tensor(pairwise_distr_np[joint_key], dtype=tf.float32)
            pairwise_distribution[joint_key] = tf.reshape(tensor, [1, int(tensor.shape[0]), int(tensor.shape[1]), 1])
    y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints], name='heat_map')
    flag_train = tf.placeholder(tf.bool, name='is_training')

    n_iters_tf = tf.Variable(0, trainable=False)

    lr_tf = tf.train.piecewise_constant(n_iters_tf, lr_decay_n_updates, lr_decay_coefs)

    # Data augmentation: we apply the same random transformations both to images and heat maps
    # x1, hm_target = tf.cond(flag_train, lambda: augmentation.augment_train(x_in, y_in), lambda: (x_in, y_in))
    x, hm_target = x_in, y_in

    # The whole heat map prediction model is here
    hm_pred_pd = model(x, n_joints, debug)
    hm_pred_pd = spatial_softmax(hm_pred_pd)

    hm_pred_sm = mrf_fixed(hm_pred_pd, pairwise_distribution)
    hm_pred_sm = spatial_softmax(hm_pred_sm)

    with tf.name_scope('loss'):
        mse_pd = mean_squared_error(hm_pred_pd, hm_target)
        mse_sm = mean_squared_error(hm_pred_sm, hm_target)
        loss = mse_pd + lmbd * weight_decay(var_pattern='weights')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(lr_tf)
        elif optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
        else:
            raise Exception('wrong optimizer')
        grads_vars = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grads_vars, global_step=n_iters_tf)

    with tf.name_scope('evaluation'):
        det_rate = 0  # evaluation.detection_rate(hm_pred, hm_target, normalized_radius=10)

    tf.summary.image('input', x, 30)
    tf.summary.image('pairwise_potential_nose_lhip', pairwise_distribution['nose_lhip'], 30)
    tb.main_summaries(grads_vars, mse_pd, mse_sm, det_rate)
    tb.show_img_plus_hm(x, hm_target, joint_names, in_height, in_width, 'target')
    tb.show_img_plus_hm(x, hm_pred_pd, joint_names, in_height, in_width, 'pred_part_detector')
    tb.show_img_plus_hm(x, hm_pred_sm, joint_names, in_height, in_width, 'pred_spatial_model')

    tb_merged = tf.summary.merge_all()
    train_iters_writer = tf.summary.FileWriter(tb_train_iter)
    train_writer = tf.summary.FileWriter(tb_train)
    test_writer = tf.summary.FileWriter(tb_test)

    saver = tf.train.Saver()

gpu_options = tf.GPUOptions(visible_device_list=gpu_number, per_process_gpu_memory_fraction=gpu_memory)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    if not restore_model:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, model_path + '/' + best_model_name)

    tb.run_summary(sess, train_writer, tb_merged, 0,
                   feed_dict={x_in:       x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to],
                              flag_train: False})
    tb.run_summary(sess, test_writer, tb_merged, 0,
                   feed_dict={x_in:       x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to],
                              flag_train: False})
    test_mse_pd, test_mse_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, batch_size)
    train_mse_pd, train_mse_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, batch_size)
    print('Epoch: {:d}  test_mse_pd: {:.5f}  test_mse_sm: {:.5f}  train_mse_pd: {:.5f}  train_mse_sm: {:.5f}'.format(
            0, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))

    if train:
        global_iter = 0
        for epoch in range(1, n_epochs + 1):
            for x_train_batch, y_train_batch in get_next_batch(x_train, y_train, batch_size, shuffle=True):
                global_iter += 1
                if tb_log_iters:
                    _, summary = sess.run([train_step, tb_merged],
                                          feed_dict={x_in: x_train_batch, y_in: y_train_batch, flag_train: True})
                    train_iters_writer.add_summary(summary, global_iter)
                else:
                    sess.run(train_step, feed_dict={x_in: x_train_batch, y_in: y_train_batch, flag_train: True})

            tb.run_summary(sess, train_writer, tb_merged, epoch,
                           feed_dict={x_in:       x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to],
                                      flag_train: False})
            tb.run_summary(sess, test_writer, tb_merged, epoch,
                           feed_dict={x_in:       x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to],
                                      flag_train: False})

            test_mse_pd, test_mse_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, batch_size)
            train_mse_pd, train_mse_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, batch_size)
            print(
                    'Epoch: {:d}  test_mse_pd: {:.5f}  test_mse_sm: {:.5f}  train_mse_pd: {:.5f}  train_mse_sm: {:.5f}'.format(
                            0, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver.save(sess, model_path + '/' + cur_timestamp)

    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))

# TODO: get first results with part-detector only and paste them into the report

# TODO: Apply special initalization to the conv weights of PGM part, based on the histogram of joint displacements!
# TODO: set up PGM part: first try the star model without trainable weight, and then try the trainable MRF


# Things that are not super important
# TODO: data: handle multiple people by incorporating an extra "torso-joint heatmap" (page 6)
# TODO: set up the option to continue training (since we should do it in 3 stages according to the paper)
# TODO: local contrast normalization: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf and
# https://github.com/bowenbaker/metaqnn/blob/master/libs/input_modules/preprocessing.py
# TODO: Data Augm.: try zero padding and random crops!
# TODO: maybe add 1x1 conv layer in the end?
# TODO: spatial dropout

# Do for the final submission
# TODO: a readme on how to run our code (1: download FLIC dataset, 2: data.py, 3: pariwise_distr.py, ...)
