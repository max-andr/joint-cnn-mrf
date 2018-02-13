import numpy as np
import os
import tensorflow as tf
import time
import pickle

import augmentation as augm
import evaluation
import tensorboard as tb
from datetime import datetime


np.set_printoptions(suppress=True, precision=4)
joint_names = np.array(['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose', 'torso'])

# star like model with chain-like inference (just to test)
# joint_dependence = {'lsho': ['nose', 'lelb'], 'lelb': ['lsho', 'lwri'], 'lwri': ['lelb'],
#                     'rsho': ['nose', 'relb'], 'relb': ['rsho', 'rwri'], 'rwri': ['relb'],
#                     'lhip': ['nose'], 'rhip': ['nose'], 'nose': ['lsho', 'rsho', 'lhip', 'rhip']}
joint_dependence = {}  # Fully-connected PGM
for joint in joint_names:
    joint_dependence[joint] = [joint_cond for joint_cond in joint_names if joint_cond != joint]


def model(x, n_joints):
    """
    A computational graph for CNN part.
    :param x1: full resolution image batch_size x 360 x 240 x 3
    :param x2: half resolution image batch_size x 180 x 120 x 3
    :return: predicted heat map batch_size x 90 x 60 x n_joints
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    n_filters = np.array([64, 128, 256, 512, 512])
    # n_filters = np.array([128, 128, 128, 512, 512])
    if debug:
        n_filters = n_filters // 4

    x1 = x
    x1 = conv_layer(x1, 5, 2, 3, n_filters[0], 'conv1_fullres')  # result: 360x240
    x1 = max_pool_layer(x1, 2, 2)  # result: 180x120
    x1 = conv_layer(x1, 5, 1, n_filters[0], n_filters[1], 'conv2_fullres')  # result: 180x120
    x1 = max_pool_layer(x1, 2, 2)  # result: 180x60
    x1 = conv_layer(x1, 5, 1, n_filters[1], n_filters[2], 'conv3_fullres')  # result: 90x60
    x1 = conv_layer(x1, 9, 1, n_filters[2], n_filters[3], 'conv4_fullres')  # result: 90x60

    x2 = tf.image.resize_images(x, [int(x.shape[1]) // 2, int(x.shape[2]) // 2])
    x2 = conv_layer(x2, 5, 2, 3, n_filters[0], 'conv1_halfres')  # result: 180x120
    x2 = max_pool_layer(x2, 2, 2)  # result: 90x60
    x2 = conv_layer(x2, 5, 1, n_filters[0], n_filters[1], 'conv2_halfres')  # result: 90x60
    x2 = max_pool_layer(x2, 2, 2)  # result: 45x30
    x2 = conv_layer(x2, 5, 1, n_filters[1], n_filters[2], 'conv3_halfres')  # result: 45x30
    x2 = conv_layer(x2, 9, 1, n_filters[2], n_filters[3], 'conv4_halfres')  # result: 45x30
    x2 = tf.image.resize_images(x2, [int(x1.shape[1]), int(x1.shape[2])])

    x3 = tf.image.resize_images(x, [int(x.shape[1]) // 4, int(x.shape[2]) // 4])
    x3 = conv_layer(x3, 5, 2, 3, n_filters[0], 'conv1_quarterres')  # result: 90x60
    x3 = max_pool_layer(x3, 2, 2)  # result: 45x30
    x3 = conv_layer(x3, 5, 1, n_filters[0], n_filters[1], 'conv2_quarterres')  # result: 45x30
    x3 = max_pool_layer(x3, 2, 2)  # result: 23x15
    x3 = conv_layer(x3, 5, 1, n_filters[1], n_filters[2], 'conv3_quarterres')  # result: 23x15
    x3 = conv_layer(x3, 9, 1, n_filters[2], n_filters[3], 'conv4_quarterres')  # result: 23x15
    x3 = tf.image.resize_images(x3, [int(x1.shape[1]), int(x1.shape[2])])

    x = x1 + x2 + x3
    x /= 3  # to compensate for summing up - should improve the convergence
    # x = tf.concat([x1, x2, x3], axis=3)
    # x = conv_layer(x, 9, 1, 3*n_filters[3], n_filters[4], 'conv5')  # result: 90x60
    x = conv_layer(x, 9, 1, n_filters[3], n_filters[4], 'conv5')  # result: 90x60
    x = conv_layer(x, 9, 1, n_filters[4], n_joints, 'conv6', last_layer=True)  # result: 90x60

    return x


def conv_mrf(A, B):
    """
    :param A: conv kernel 1 x 120 x 180 x 1 (prior)
    :param B: input heatmaps: batch_size x 60 x 90 x 1 (likelihood)
    :return: C is batch_size x 60 x 90 x 1
    """
    B = tf.transpose(B, [1, 2, 3, 0])  # tf.reshape(B, [hm_height, hm_width, 1, tf.shape(B)[0]])
    B = tf.reverse(B, axis=[0, 1])  # we flip kernel to get convolution

    # conv between 1 x 120 x 180 x 1 and 60 x 90 x 1 x ? => 1 x 61 x 91 x ?
    C = tf.nn.conv2d(A, B, strides=[1, 1, 1, 1], padding='VALID')  # 1 x 61 x 91 x ?
    C = C[:, :hm_height, :hm_width, :]  # 1 x 60 x 90 x ?
    C = tf.transpose(C, [3, 1, 2, 0])  # tf.reshape(C, [tf.shape(B)[3], hm_height, hm_width, 1])
    return C


def spatial_model(heat_map):
    """
    from Learning Human Pose Estimation Features with Convolutional Networks
    :param heat_map: is produced by model as the unary distributions: batch_size x 90 x 60 x n_joints
    :param pairwise_energies: the priors for a pair of joints, it's calculated from the histogram and is fixed 1x180x120xn_pairs
    :return heat_map_hat: the result from equation 1 in the paper: batch_size x 90 x 60 x n_joints
    """

    def relu_pos(x, eps=0.00001):
        return tf.maximum(x, eps)

    def softplus(x):
        softplus_alpha = 5
        return 1 / softplus_alpha * tf.nn.softplus(softplus_alpha * x)

    delta = 10 ** -6  # for numerical stability
    heat_map_hat = []
    for joint_id, joint_name in enumerate(joint_names[:n_joints]):
        with tf.variable_scope(joint_name):
            hm = heat_map[:, :, :, joint_id:joint_id + 1]
            hm = batch_norm(hm)
            marginal_energy = tf.log(softplus(hm))  # heat_map: batch_size x 90 x 60 x 1
            for cond_joint in joint_dependence[joint_name]:
                cond_joint_id = np.where(joint_names == cond_joint)[0][0]
                prior = softplus(pairwise_energies[joint_name + '_' + cond_joint])
                likelihood = heat_map[:, :, :, cond_joint_id:cond_joint_id + 1]
                bias = softplus(pairwise_biases[joint_name + '_' + cond_joint])

                marginal_energy += tf.log(conv_mrf(prior, likelihood) + bias + delta)
            heat_map_hat.append(marginal_energy)
    return tf.stack(heat_map_hat, axis=3)[:, :, :, :, 0]


def batch_norm(x):
    x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=flag_train)
    return x


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
    # print(shape)
    return tf.get_variable('weights', initializer=initial)


def bias_variable(shape, init=0.0, name='biases'):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(init, shape=shape)
    return tf.get_variable(name, initializer=initial)


def conv_layer(x, size, stride, n_in, n_out, name, last_layer=False):
    with tf.variable_scope(name):
        w = weight_variable([size, size, n_in, n_out])
        b = bias_variable([n_out])
        pre_activ = conv2d(x, w, stride) + b
        if not last_layer:
            activ = tf.nn.relu(pre_activ)
            activ = batch_norm(activ)
        else:
            activ = pre_activ
    # we do it out of the namescope to show it separately in Tensorboard
    tb.var_summary(pre_activ, name)
    tf.summary.image('f_activ_' + name, activ[:, :, :, 7:8], 3)
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
        x_idx = np.arange(len(X))[:n_batches * batch_size]
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


def spatial_softmax(hm):
    hm_height, hm_width, n_joints = int(hm.shape[1]), int(hm.shape[2]), int(hm.shape[3])
    hm = tf.reshape(hm, [-1, hm_height * hm_width, n_joints])
    hm = tf.nn.softmax(logits=hm, dim=1)
    hm = tf.reshape(hm, [-1, hm_height, hm_width, n_joints])
    return hm


def cross_entropy(hm1, hm2):
    """
    Mean squared error between 2 heat maps for a batch as described in
    [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

    hm1, hm2: tensor of size [n_images, height, width, n_joints]
    """
    # if we don't multiply by number of pixels, then we get too small value of the loss
    # return tf.reduce_mean((hm1 - hm2) ** 2) * hm_height * hm_width * n_joints
    hm_height, hm_width, n_joints = int(hm1.shape[1]), int(hm1.shape[2]), int(hm1.shape[3])
    hm1 = tf.reshape(hm1, [-1, hm_height*hm_width, n_joints])
    hm2 = tf.reshape(hm2, [-1, hm_height*hm_width, n_joints])
    # loss_list = []
    # for i in range(n_joints):
    #     loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=hm1[:, :, i], labels=hm2[:, :, i])
    #     loss_list.append(loss_i)
    # loss = tf.stack(loss_list, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=hm1, labels=hm2, dim=1)
    return tf.reduce_mean(loss)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def avg_tensor_list(tensor_list):
    tensors = tf.stack(axis=0, values=tensor_list)
    return tf.reduce_mean(tensors, axis=0)


def eval_error(X_np, Y_np, sess, batch_size):
    """Get all predictions for a dataset by running it in small batches."""
    n_batches = len(X_np) // batch_size
    mse_pd_val, mse_sm_val, det_rate_pd_val, det_rate_sm_val = 0.0, 0.0, 0.0, 0.0
    for batch_x, batch_y in get_next_batch(X_np, Y_np, batch_size):
        v1, v2, v3, v4 = sess.run([loss_pd, loss_sm, det_rate_pd, det_rate_sm], feed_dict={x_in: batch_x, y_in: batch_y, flag_train: False})
        mse_pd_val, mse_sm_val = mse_pd_val + v1, mse_sm_val + v2
        det_rate_pd_val, det_rate_sm_val = det_rate_pd_val + v3, det_rate_sm_val + v4
    return mse_pd_val / n_batches, mse_sm_val / n_batches, det_rate_pd_val / n_batches, det_rate_sm_val / n_batches


def get_dataset():
    x_train = np.load('x_train_flic.npy')
    x_test = np.load('x_test_flic.npy')
    y_train = np.load('y_train_flic.npy')
    y_test = np.load('y_test_flic.npy')
    return x_train, y_train, x_test, y_test


def get_pairwise_distr():
    with open('pairwise_distribution.pickle', 'rb') as handle:
        return pickle.load(handle)


debug = False
multi_gpu = not debug
data_augm = True  # not debug
use_sm = True
if multi_gpu:
    # gpus, gpu_memory = [0, 1, 2, 3, 4, 5, 6, 7], 0.4
    gpus, gpu_memory = [0, 1, 2, 3], 0.5
else:
    gpus, gpu_memory = [6], 0.7
train, restore_model, best_model_name = True, False, '2018-02-09 13:18:03'
model_path = 'models_ex'
time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
tb_folder = 'tb'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, cur_timestamp)
tb_train = '{}/{}/train'.format(tb_folder, cur_timestamp)
tb_test = '{}/{}/test'.format(tb_folder, cur_timestamp)
tb_log_iters = False
n_eval_ex = 512 if debug else 1100

n_joints = 9  # excluding "torso-joint", which is 10-th
x_train, y_train, x_test, y_test = get_dataset()
pairwise_distr_np = get_pairwise_distr()
n_train, in_height, in_width, n_colors = x_train.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
if debug:
    n_train, n_test = 1024, 512  # for debugging purposes we take only a small subset
    train_idx, test_idx = np.random.permutation(n_train), np.random.permutation(n_test)
    x_train, y_train, x_test, y_test = x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]
# Main hyperparameters
n_epochs = 30 if debug else 60
batch_size = 16
lmbd = 0.0001  # best: 0.1 for debug, and 0.0001
lr, optimizer = 0.001, 'adam'  # best: 0.001, 'adam'
n_updates_total = n_epochs * n_train // batch_size
lr_decay_n_updates = [round(0.7 * n_updates_total), round(0.8 * n_updates_total), round(0.9 * n_updates_total)]
lr_decay_coefs = [lr, lr / 2, lr / 5, lr / 10]
img_tb_from = 70  # 50 or 450
img_tb_to = img_tb_from + batch_size

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    x_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
    if use_sm:
        pairwise_energies, pairwise_biases = {}, {}
        for joint in joint_names[:n_joints]:
            for cond_joint in joint_dependence[joint]:
                joint_key = joint + '_' + cond_joint
                tensor = tf.convert_to_tensor(pairwise_distr_np[joint_key], dtype=tf.float32)
                pairwise_energy_jj = tf.reshape(tensor, [1, tensor.shape[0].value, tensor.shape[1].value, 1])
                pairwise_energies[joint_key] = tf.get_variable('energy_' + joint_key, initializer=pairwise_energy_jj)
                pairwise_biases[joint_key] = bias_variable([1, hm_height, hm_width, 1], 0.0001, 'bias_' + joint_key)
    y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints+1], name='heat_map')
    flag_train = tf.placeholder(tf.bool, name='is_training')

    n_iters_tf = tf.get_variable('n_iters', initializer=0, trainable=False)
    lr_tf = tf.train.piecewise_constant(n_iters_tf, lr_decay_n_updates, lr_decay_coefs)

    # Data augmentation: we apply the same random transformations both to images and heat maps
    if data_augm:
        x_batch, hm_target_batch = tf.cond(flag_train, lambda: augm.augment_train(x_in, y_in),
                                           lambda: augm.augment_test(x_in, y_in))
    else:
        x_batch, hm_target_batch = x_in, y_in

    if optimizer == 'adam':
        opt = tf.train.AdamOptimizer(lr_tf)
    elif optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
    else:
        raise Exception('wrong optimizer')

    # Calculate the gradients for each model tower.
    tower_grads, losses, det_rates_pd, det_rates_sm, hms_pred_pd, hms_pred_sm = [], [], [], [], [], []
    mses_pd, mses_sm = [], []
    imgs_per_gpu = batch_size // len(gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(gpus)):
            with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:
                # Dequeues one batch for the GPU
                id_from, id_to = i*imgs_per_gpu, i*imgs_per_gpu + imgs_per_gpu
                x, hm_target = x_batch[id_from:id_to], hm_target_batch[id_from:id_to]
                # Calculate the loss for one tower of the CIFAR model. This function
                # constructs the entire CIFAR model but shares the variables across all towers.

                # The whole heat map prediction model is here
                hm_pred_pd_logit = model(x, n_joints)
                hm_pred_pd = spatial_softmax(hm_pred_pd_logit)
                hms_pred_pd.append(hm_pred_pd)

                if use_sm:
                    # To disambiguate multiple people on the same image
                    hm_pred_pd_with_torso = tf.concat([hm_pred_pd, hm_target[:, :, :, n_joints:]], axis=3)

                    hm_pred_sm_logit = spatial_model(hm_pred_pd_with_torso)
                    hm_pred_sm = spatial_softmax(hm_pred_sm_logit)
                    hms_pred_sm.append(hm_pred_sm)
                else:
                    # for compatibility with Tensorboard, we should have this variable
                    hm_pred_sm_logit, hm_pred_sm = hm_pred_pd_logit, hm_pred_pd
                    hms_pred_sm.append(hm_pred_sm)

                with tf.name_scope('loss'):
                    loss_pd = cross_entropy(hm_pred_pd_logit, hm_target[:, :, :, :n_joints])
                    loss_sm = cross_entropy(hm_pred_sm_logit, hm_target[:, :, :, :n_joints])
                    loss_tower = loss_pd + loss_sm + lmbd * weight_decay(var_pattern='weights')
                    losses.append(loss_tower)
                    mses_pd.append(loss_tower)
                    mses_sm.append(loss_tower)

                with tf.name_scope('evaluation'):
                    joints_to_eval = [2, 5]  # 'all'
                    wrist_det_rate10_pd = evaluation.det_rate(hm_pred_pd, hm_target[:, :, :, :n_joints], normalized_radius=10, joints=joints_to_eval)
                    wrist_det_rate10_sm = evaluation.det_rate(hm_pred_sm, hm_target[:, :, :, :n_joints], normalized_radius=10, joints=joints_to_eval)
                    det_rates_pd.append(wrist_det_rate10_pd)
                    det_rates_sm.append(wrist_det_rate10_sm)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Calculate the gradients for the batch of data on this tower.
                    grads_vars_in_tower = opt.compute_gradients(loss_tower)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads_vars_in_tower)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads_vars = average_gradients(tower_grads)
    loss = avg_tensor_list(losses)
    det_rate_pd = avg_tensor_list(det_rates_pd)
    det_rate_sm = avg_tensor_list(det_rates_sm)
    mses_pd = avg_tensor_list(mses_pd)
    mses_sm = avg_tensor_list(mses_sm)
    hms_pred_pd = tf.concat(hms_pred_pd, axis=0)
    hms_pred_sm = tf.concat(hms_pred_sm, axis=0)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=n_iters_tf)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.99, n_iters_tf)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_step = tf.group(apply_gradient_op, variables_averages_op)

    tf.summary.image('input', x_batch, 30)
    if use_sm:
        for key in pairwise_energies:
            tf.summary.image('pairwise_potential_' + key, pairwise_energies[key], 30)
            tf.summary.image('pairwise_biases_' + key, pairwise_biases[key], 30)
            tb.var_summary(pairwise_energies[key], 'pairwise_energies_' + key)
            tb.var_summary(pairwise_biases[key], 'pairwise_biases_' + key)
    tb.main_summaries(grads_vars)
    tb.show_img_plus_hm(x_batch, hm_target_batch[:, :, :, :n_joints], joint_names[:n_joints], in_height, in_width, 'target')
    tb.show_img_plus_hm(x_batch, hms_pred_pd, joint_names[:n_joints], in_height, in_width, 'pred_part_detector')
    tb.show_img_plus_hm(x_batch, hms_pred_sm, joint_names[:n_joints], in_height, in_width, 'pred_spatial_model')

    tb_merged = tf.summary.merge_all()
    train_iters_writer = tf.summary.FileWriter(tb_train_iter)
    train_writer = tf.summary.FileWriter(tb_train)
    test_writer = tf.summary.FileWriter(tb_test)

    saver = tf.train.Saver()

gpu_options = tf.GPUOptions(visible_device_list=str(gpus)[1:-1], per_process_gpu_memory_fraction=gpu_memory)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
with tf.Session(config=config, graph=graph) as sess:
    if not restore_model:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, model_path + '/' + best_model_name)
        sess.run(tf.variables_initializer([n_iters_tf]))

    tb.run_summary(sess, train_writer, tb_merged, 0,
                   feed_dict={x_in:       x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to],
                              flag_train: False})
    tb.run_summary(sess, test_writer, tb_merged, 0,
                   feed_dict={x_in:       x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to],
                              flag_train: False})
    train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, batch_size)
    test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, batch_size)
    print('Epoch {:d}  test_dr {:.3f} {:.3f}  train_dr {:.3f} {:.3f}  test_mse {:.5f} {:.5f}  train_mse {:.5f} {:.5f}'.
        format(0, test_dr_pd, test_dr_sm, train_dr_pd, train_dr_sm, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))
    tb.write_summary(test_writer, [test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm],
                     ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], 0)
    tb.write_summary(train_writer, [train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm],
                     ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], 0)
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

            tb.run_summary(sess, train_writer, tb_merged, epoch, feed_dict={
                x_in: x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to], flag_train: False})
            tb.run_summary(sess, test_writer, tb_merged, epoch, feed_dict={
                x_in: x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to], flag_train: False})
            train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, batch_size)
            test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, batch_size)
            print('Epoch {:d}  test_dr {:.3f} {:.3f}  train_dr {:.3f} {:.3f}  test_mse {:.5f} {:.5f}  train_mse {:.5f} {:.5f}'.
                  format(epoch, test_dr_pd, test_dr_sm, train_dr_pd, train_dr_sm, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))
            tb.write_summary(test_writer, [test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm],
                             ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], epoch)
            tb.write_summary(train_writer, [train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm],
                             ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], epoch)
            # Save the model on each epoch after half of epochs are done
            if epoch > n_epochs // 2:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, '{}/{}_epoch{}'.format(model_path, cur_timestamp, epoch))

    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))

# TODO: show a principled plot of test MSE between PD and SM.

# Do for the final submission
# TODO: a readme on how to run our code (1: download FLIC dataset, 2: data.py, 3: pariwise_distr.py, ...)

"""
Structure:
- Problem description
- Your approach
- Results / Interpretation

Idea: CNNs are great, but can't be easily controlled. We need to impose strong prior knowledge on the 
relations between parts.


To mention in the final report:
- we do it much faster using BN
- we use an auxiliary classifier (Inception style or like in Fast R-CNN: "multi-task loss") to make both PD and SM perform well
This goes in line with motivation from the paper.
- thus we can train the model end-to-end from scratch and much faster (30 minutes instead of 60 hours)
but we should train on FLIC+...
- we provide understanding on what the model learns
- we fix the mistakes from the paper and fill the gaps (leaves mixed feeling, how could it be that they mixed up
the dimensions of the convolutions; if you really train model X, you just translate it to the description)
- we show how can we improve the pairwise potentials if we optimize them.

The paper leaves mixed feeling, especially because they didn't explain what their SM learn.
On Fig. 5 they showed "a didactic example", which they most probably drew by hand.
It would be very interesting to see the pairwise heatmaps obtained after backprop. We show them: ...
Thus, our contribution is not only in practical implementation of the paper, but also in understanding what the proposed 
model actually learns.

We claim that their statement "The learned pair-wise distributions are purely uniform when any pairwise edge should to
be removed from the graph structure" is wrong. Even connections that don't appear in a traditional star model are not 
uniform. E.g. relation between lhip and rwri. Of course, their relative positions can vary a lot, but there are certainly 
regions that have 0 probability (e.g. too far away): *show picture*

Smart init makes sense: test_dr 0.2% 7.1% with random weights


WD only over conv filters, since we observed that the highest values of pairwise potentials and biases are
quite moderate, so it doesn't make sense to include WD there.

Another contribution: we show how to perform joint training immediately with a single set of hps.

softplus -> relu? no motivation why softplus is used

Discuss the magnitude of grads wrt pairwise energies/biases (should be small => training is successful).

Discuss that conv filters are like edge/color detectors.

Cross entropy: much faster convergence.

Eldar et. al (DeepCut): Hyperparameter search is very important (especially if the hps were not reported!).

ResNets paper: use downsampling with the stride=2 in the beginning (unlike in the paper, we don't lose the information!).

More advanced DA.

SM takes half of the time needed for PD.

Show evolution of pairwise potentials over iters (select from those that are arleady lwri|lelb. however, the role of
pairwise pots is slightly less useful for LSP)


We excluded self connections like face|face

Boring but important implementation detail: BN is not so efficient in multi-gpu training. 

With Momentum the spatial model can't be properly trained. Thus adaptive learning rates (Adam).

Interesting Q: how detection rate correlates with MSE?
"""
