import argparse
import numpy as np
import os
import tensorflow as tf
import time
import pickle
import scipy.io
import skimage.transform
import matplotlib.pyplot as plt

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
    A computational graph for the part detector (CNN). Note that the size of heat maps is 8 times smaller (due to
    a convolution with stride=2 and 2 max pooling layers) than the original image. However, even such huge downsampling
    preserves satisfactory localization, and significantly saves computational power.
    :param x: full resolution image hps.batch_size x 480 x 720 x 3
    :return: predicted heat map hps.batch_size x 60 x 90 x n_joints.
    """
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    n_filters = np.array([64, 128, 256, 512, 512])
    # n_filters = np.array([128, 128, 128, 512, 512])
    if hps.debug:
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
    x = conv_layer(x, 9, 1, n_filters[3], n_filters[4], 'conv5')  # result: 90x60
    x = conv_layer(x, 9, 1, n_filters[4], n_joints, 'conv6', last_layer=True)  # result: 90x60

    return x


def conv_mrf(A, B):
    """
    :param A: conv kernel 1 x 120 x 180 x 1 (prior)
    :param B: input heatmaps: hps.batch_size x 60 x 90 x 1 (likelihood)
    :return: C is hps.batch_size x 60 x 90 x 1
    """
    B = tf.transpose(B, [1, 2, 3, 0])
    B = tf.reverse(B, axis=[0, 1])  # [h, w, 1, b], we flip kernel to get convolution, and not cross-correlation

    # conv between 1 x 120 x 180 x 1 and 60 x 90 x 1 x ? => 1 x 61 x 91 x ?
    C = tf.nn.conv2d(A, B, strides=[1, 1, 1, 1], padding='VALID')  # 1 x 61 x 91 x ?
    # C = C[:, :hm_height, :hm_width, :]  # 1 x 60 x 90 x ?
    C = tf.image.resize_images(C, [hm_height, hm_width])
    C = tf.transpose(C, [3, 1, 2, 0])
    return C


def spatial_model(heat_map):
    """
    Implementation of the spatial model in log space (given by Eq. 2 in the original paper).
    :param heat_map: is produced by model as the unary distributions: hps.batch_size x 60 x 90 x n_joints
    """

    def relu_pos(x, eps=0.00001):
        """
        It is described in the paper, but we decided not to use it. Instead we apply softplus everywhere.
        """
        return tf.maximum(x, eps)

    def softplus(x):
        softplus_alpha = 5
        return 1 / softplus_alpha * tf.nn.softplus(softplus_alpha * x)

    delta = 10 ** -6  # for numerical stability
    heat_map_hat = []
    with tf.variable_scope('bn_sm'):
        heat_map = tf.contrib.layers.batch_norm(heat_map, decay=0.9, center=True, scale=True, is_training=flag_train)
    for joint_id, joint_name in enumerate(joint_names[:n_joints]):
        with tf.variable_scope(joint_name):
            hm = heat_map[:, :, :, joint_id:joint_id + 1]
            marginal_energy = tf.log(softplus(hm) + delta)  # heat_map: batch_size x 90 x 60 x 1
            for cond_joint in joint_dependence[joint_name]:
                cond_joint_id = np.where(joint_names == cond_joint)[0][0]
                prior = softplus(pairwise_energies[joint_name + '_' + cond_joint])
                likelihood = softplus(heat_map[:, :, :, cond_joint_id:cond_joint_id + 1])
                bias = softplus(pairwise_biases[joint_name + '_' + cond_joint])
                marginal_energy += tf.log(conv_mrf(prior, likelihood) + bias + delta)
            heat_map_hat.append(marginal_energy)
    return tf.stack(heat_map_hat, axis=3)[:, :, :, :, 0]


def batch_norm(x):
    x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=flag_train, trainable=train_pd)
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
    return tf.get_variable('weights', initializer=initial, trainable=train_pd)


def bias_variable(shape, init=0.0, name='biases'):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(init, shape=shape)
    return tf.get_variable(name, initializer=initial, trainable=train_pd)


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
        x_idx = np.random.permutation(len(X))[:n_batches * hps.batch_size]
    else:
        x_idx = np.arange(len(X))[:n_batches * hps.batch_size]
    for batch_idx in x_idx.reshape([n_batches, hps.batch_size]):
        batch_x, batch_y = X[batch_idx], Y[batch_idx]
        yield batch_x, batch_y


def weight_decay(var_pattern):
    """
    L2 weight decay loss, based on all weights that have var_pattern in their name

    var_pattern - a substring of a name of weights variables that we want to use in Weight Decay.
    """
    costs = []
    for var in tf.global_variables():
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


def softmax_cross_entropy(hm1, hm2):
    """
    Softmax applied over 2 spatial dimensions (for this we do reshape) followed by cross-entropy.

    hm1, hm2: tensor of size [n_images, height, width, n_joints]
    """
    # MSE: tf.reduce_mean((hm1 - hm2) ** 2) * hm_height * hm_width * n_joints
    hm_height, hm_width, n_joints = int(hm1.shape[1]), int(hm1.shape[2]), int(hm1.shape[3])
    hm1 = tf.reshape(hm1, [-1, hm_height*hm_width, n_joints])
    hm2 = tf.reshape(hm2, [-1, hm_height*hm_width, n_joints])

    # Element-wise sigmoid with binary cross-entropy on top of them:
    # loss_list = []
    # for i in range(n_joints):
    #     loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=hm1[:, :, i], labels=hm2[:, :, i])
    #     loss_list.append(loss_i)
    # loss = tf.stack(loss_list, axis=1)

    # Our choice: softmax applied over 2 spatial dimensions followed by cross-entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=hm1, labels=hm2, dim=1)
    return tf.reduce_mean(loss)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
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
    n_batches = len(X_np) // hps.batch_size
    mse_pd_val, mse_sm_val, det_rate_pd_val, det_rate_sm_val = 0.0, 0.0, 0.0, 0.0
    for batch_x, batch_y in get_next_batch(X_np, Y_np, hps.batch_size):
        v1, v2, v3, v4 = sess.run([loss_pd, loss_sm, det_rate_pd, det_rate_sm], feed_dict={x_in: batch_x, y_in: batch_y, flag_train: False})
        mse_pd_val, mse_sm_val = mse_pd_val + v1, mse_sm_val + v2
        det_rate_pd_val, det_rate_sm_val = det_rate_pd_val + v3, det_rate_sm_val + v4
    return mse_pd_val / n_batches, mse_sm_val / n_batches, det_rate_pd_val / n_batches, det_rate_sm_val / n_batches


def get_dataset():
    """
    Note, that in order to have these files you need to run `data.py` first.
    """
    x_train = np.load('x_train_flic.npy')
    x_test = np.load('x_test_flic.npy')
    y_train = np.load('y_train_flic.npy')
    y_test = np.load('y_test_flic.npy')
    return x_train, y_train, x_test, y_test


def get_pairwise_distr():
    with open('pairwise_distribution.pickle', 'rb') as handle:
        return pickle.load(handle)


def grad_renorm(gs_vs, norm):
    """
    It is useful to stabilize the training, especially with high amount of weight decay.
    """
    grads, vars = zip(*gs_vs)
    grads, _ = tf.clip_by_global_norm(grads, norm)
    gs_vs = zip(grads, vars)
    return gs_vs


def get_gpu_memory(gpus):
    """
    Small heuristic to calculate the amount of memory needed for each GPU in case of multi-gpu training.
    """
    if len(gpus) >= 5:
        return 0.4
    elif len(gpus) >= 3:
        return 0.5
    elif len(gpus) == 2:
        return 0.6
    else:
        return 0.6


def get_different_scales(x, pad_array, crop_array, orig_h, orig_w):
    x_new = []
    for pad_c in pad_array:
        n_pad_h = round(orig_h * (pad_c - 1) / 2)
        n_pad_w = round(orig_w * (pad_c - 1) / 2)
        x_pad = np.lib.pad(x, ((n_pad_h, n_pad_h), (n_pad_w, n_pad_w), (0, 0)), 'constant', constant_values=0)
        x_orig_size = skimage.transform.resize(x_pad, (orig_h, orig_w))
        x_new.append(x_orig_size)
    for crop_c in crop_array:
        h1 = round((1-crop_c)/2*orig_h)
        h2 = h1 + round(crop_c*orig_h)
        w1 = round((1-crop_c)/2*orig_w)
        w2 = w1 + round(crop_c*orig_w)
        x_crop = x[h1:h2, w1:w2]
        x_orig_size = skimage.transform.resize(x_crop, (orig_h, orig_w))
        x_new.append(x_orig_size)

    # for i in range(9):
    #     plt.figure(i)
    #     plt.imshow(x_new[i])
    #     plt.savefig('img/img_'+str(i)+'.png', dpi=300)
    #     plt.clf()
    return np.array(x_new)


def scale_hm_back(hms, pad_array, crop_array, orig_h, orig_w):
    hms_new = []
    for i, crop_c in enumerate(pad_array):
        crop_c = 1 / crop_c
        h1 = round((1-crop_c)/2*orig_h)
        h2 = h1 + round(crop_c*orig_h)
        w1 = round((1-crop_c)/2*orig_w)
        w2 = w1 + round(crop_c*orig_w)
        hm_crop = hms[i][h1:h2, w1:w2]
        hm_orig_size = skimage.transform.resize(hm_crop, (orig_h, orig_w))
        hms_new.append(hm_orig_size)

    for i, pad_c in enumerate(crop_array):
        pad_c = 1 / pad_c
        n_pad_h = round(orig_h * (pad_c - 1) / 2)
        n_pad_w = round(orig_w * (pad_c - 1) / 2)
        hm_pad = np.lib.pad(hms[i+len(pad_array)], ((n_pad_h, n_pad_h), (n_pad_w, n_pad_w), (0, 0)), 'constant', constant_values=0)
        hm_orig_size = skimage.transform.resize(hm_pad, (orig_h, orig_w))
        hms_new.append(hm_orig_size)

    # for i in range(9):
    #     plt.figure(i)
    #     plt.imshow(x_new[i][:, :, 8])
    #     plt.savefig('img/img_'+str(i)+'_processed.png', dpi=300)
    #     plt.clf()
    #     plt.imshow(hms[i][:, :, 8])
    #     plt.savefig('img/img_'+str(i)+'_orig.png', dpi=300)
    #     plt.clf()
    return np.array(hms_new)


def get_predictions(X_np, Y_np, sess):
    """ Get all predictions for a dataset by running it in small batches.
        We use a multi-scale evaluation procedure proposed in "Learning human pose estimation features with
        convolutional networks". We strongly suspect that this procedure was used in the original paper. However,
        they do not report it.
        Without this procedure it's impossible to reproduce their part detector.
    """
    def argmax_hm(hm):
        hm = np.squeeze(hm)
        hm = np.reshape(hm, [hm_height * hm_width, n_joints])
        coords_raw = np.argmax(hm, axis=0)  # [n_images, n_joints]
        # Now we obtain real spatial coordinates for each image and for each joint
        coords_x = coords_raw // hm_width
        coords_y = coords_raw - coords_x * hm_width
        coords_xy = np.stack([coords_x, coords_y], axis=0)
        return coords_xy

    n = 1100
    X_np, Y_np = X_np[:n], Y_np[:n]
    drs_pd, drs_sm, pred_coords_pd, pred_coords_sm = [], [], [], []
    pad_array, crop_array = [1.1, 1.2, 1.3, 1.4], [0.7, 0.8, 0.9, 1.0]
    for x_np, y_np in zip(X_np, Y_np):
        x_np_diff_scales = get_different_scales(x_np, pad_array, crop_array, in_height, in_width)
        y_np = np.repeat(np.expand_dims(y_np, 0), x_np_diff_scales.shape[0], axis=0)
        hm_pd_np, hm_sm_np = sess.run([hm_pred_pd, hm_pred_sm], feed_dict={x_in: x_np_diff_scales, y_in: y_np, flag_train: False})
        hm_pd_np = scale_hm_back(hm_pd_np, pad_array, crop_array, hm_height, hm_width)
        hm_sm_np = scale_hm_back(hm_sm_np, pad_array, crop_array, hm_height, hm_width)
        # argmax over 1st dimension to get the most confident prediction for each joint
        # hm_pd_np = np.max(hm_pd_np, axis=0, keepdims=True)
        # hm_sm_np = np.max(hm_sm_np, axis=0, keepdims=True)

        hm_pd_np = np.expand_dims(np.average(hm_pd_np, axis=0), 0)
        hm_sm_np = np.expand_dims(np.average(hm_sm_np, axis=0), 0)

        pred_coords_pd.append(argmax_hm(hm_pd_np))
        pred_coords_sm.append(argmax_hm(hm_sm_np))

        # input aggregated hm and get det_rate
        dr_pd, dr_sm = sess.run([wrist_det_rate10_pd, wrist_det_rate10_sm],
                                feed_dict={hm_pred_pd: hm_pd_np, hm_pred_sm: hm_sm_np, y_in: y_np, flag_train: False})
        drs_pd.append(dr_pd)
        drs_sm.append(dr_sm)
    print('test_dr: {} {}'.format(np.average(drs_pd), np.average(drs_sm)))
    return np.stack(pred_coords_pd, axis=2), np.stack(pred_coords_sm, axis=2)


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--debug', action='store_true', help='True if we want to debug.')
parser.add_argument('--train', action='store_true', help='True if we want to train the model.')
parser.add_argument('--gpus', nargs='+', type=int, default=[6], help='GPU indices.')
parser.add_argument('--restore', action='store_true', help='True if we want to restore the model.')
parser.add_argument('--use_sm', action='store_true', help='True if we want to use the Spatial Model.')
parser.add_argument('--data_augm', action='store_true', help='True if we want to use data augmentation.')
parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=14, help='Batch size.')
parser.add_argument('--optimizer', type=str, default='adam', help='momentum or adam')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--lmbd', type=float, default=0.001, help='Regularization coefficient.')
hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10

gpu_memory = get_gpu_memory(hps.gpus)
train_pd = True  # to train the part detector or not
best_model_name = '2018-02-17 11:34:12_lr=0.001_lambda=0.001_bs=14-76'
model_path = 'models_ex'
time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
model_name = '{}_lr={}_lambda={}_bs={}'.format(cur_timestamp, hps.lr, hps.lmbd, hps.batch_size)
tb_folder = 'tb'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, model_name)  # is not used, but can be useful for debugging
tb_train = '{}/{}/train'.format(tb_folder, model_name)
tb_test = '{}/{}/test'.format(tb_folder, model_name)
tb_log_iters = False
n_eval_ex = 512 if hps.debug else 1100
joints_to_eval = [2]  # for example, 2 - left wrist, 5 - right wrist, 8 - nose, 'all' - all joints
det_radius = 10  # moderate detection radius

n_joints = 9  # excluding "torso-joint", which is 10-th
x_train, y_train, x_test, y_test = get_dataset()
pairwise_distr_np = get_pairwise_distr()
n_train, in_height, in_width, n_colors = x_train.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
if hps.debug:
    n_train, n_test = 1024, 512  # for debugging purposes we take only a small subset
    train_idx, test_idx = np.random.permutation(x_train.shape[0])[:n_train], np.random.permutation(x_test.shape[0])[:n_test]
    x_train, y_train, x_test, y_test = x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]
# Main hyperparameters
n_updates_total = hps.n_epochs * n_train // hps.batch_size
lr_decay_n_updates = [round(0.7 * n_updates_total), round(0.8 * n_updates_total), round(0.9 * n_updates_total)]
lr_decay_coefs = [hps.lr, hps.lr / 2, hps.lr / 5, hps.lr / 10]
img_tb_from = 450  # 50 or 450
img_tb_to = img_tb_from + hps.batch_size

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    x_in = tf.placeholder(tf.float32, [None, in_height, in_width, n_colors], name='input_full')
    if hps.use_sm:
        pairwise_energies, pairwise_biases = {}, {}
        for joint in joint_names[:n_joints]:
            for cond_joint in joint_dependence[joint]:
                joint_key = joint + '_' + cond_joint
                tensor = tf.convert_to_tensor(pairwise_distr_np[joint_key], dtype=tf.float32)
                pairwise_energy_jj = tf.reshape(tensor, [1, tensor.shape[0].value, tensor.shape[1].value, 1])
                pairwise_energies[joint_key] = tf.get_variable('energy_' + joint_key, initializer=pairwise_energy_jj)

                init = tf.constant(0.00001, shape=[1, hm_height, hm_width, 1])
                pairwise_biases[joint_key] = tf.get_variable('bias_' + joint_key, initializer=init)
    y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints+1], name='heat_map')
    flag_train = tf.placeholder(tf.bool, name='is_training')

    n_iters_tf = tf.get_variable('n_iters', initializer=0, trainable=False)
    lr_tf = tf.train.piecewise_constant(n_iters_tf, lr_decay_n_updates, lr_decay_coefs)

    # Data augmentation: we apply the same random transformations both to images and heat maps
    if hps.data_augm:
        x_batch, hm_target_batch = tf.cond(flag_train, lambda: augm.augment_train(x_in, y_in),
                                           lambda: augm.augment_test(x_in, y_in))
    else:
        x_batch, hm_target_batch = x_in, y_in

    if hps.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(lr_tf)
    elif hps.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9)
    else:
        raise Exception('wrong optimizer')

    # Calculate the gradients for each model tower.
    tower_grads, losses, det_rates_pd, det_rates_sm, hms_pred_pd, hms_pred_sm = [], [], [], [], [], []
    mses_pd, mses_sm = [], []
    imgs_per_gpu = hps.batch_size // len(hps.gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(hps.gpus)):
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

                if hps.use_sm:
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
                    loss_pd = softmax_cross_entropy(hm_pred_pd_logit, hm_target[:, :, :, :n_joints])
                    loss_sm = softmax_cross_entropy(hm_pred_sm_logit, hm_target[:, :, :, :n_joints])
                    loss_tower = loss_pd + loss_sm + hps.lmbd * weight_decay(var_pattern='weights')
                    losses.append(loss_tower)
                    mses_pd.append(loss_tower)
                    mses_sm.append(loss_tower)

                with tf.name_scope('evaluation'):
                    wrist_det_rate10_pd = evaluation.det_rate(hm_pred_pd, hm_target[:, :, :, :n_joints],
                                                              normalized_radius=det_radius, joints=joints_to_eval)
                    wrist_det_rate10_sm = evaluation.det_rate(hm_pred_sm, hm_target[:, :, :, :n_joints],
                                                              normalized_radius=det_radius, joints=joints_to_eval)
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

    grads_vars = grad_renorm(grads_vars, 4.0)
    train_step = opt.apply_gradients(grads_vars, global_step=n_iters_tf)

    # # Track the moving averages of all trainable variables. We did not use it in the final evaluation.
    # variable_averages = tf.train.ExponentialMovingAverage(0.99, n_iters_tf)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #
    # # Group all updates to into a single train op.
    # train_step = tf.group(apply_gradient_op, variables_averages_op)

    tf.summary.image('input', x_batch, 30)
    if hps.use_sm:
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
    train_iters_writer = tf.summary.FileWriter(tb_train_iter, flush_secs=30)
    train_writer = tf.summary.FileWriter(tb_train, flush_secs=30)
    test_writer = tf.summary.FileWriter(tb_test, flush_secs=30)

    # saver_old = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'bn_sm' not in v.name])
    saver = tf.train.Saver(max_to_keep=50)

gpu_options = tf.GPUOptions(visible_device_list=str(hps.gpus)[1:-1], per_process_gpu_memory_fraction=gpu_memory)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
with tf.Session(config=config, graph=graph) as sess:
    if not hps.restore:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, model_path + '/' + best_model_name)
        # vars_to_init = [var for var in tf.global_variables() if 'energy' in var.op.name or 'bias_' in var.op.name]
        # vars_to_init = [var for var in tf.global_variables() if 'bn_sm' in var.op.name]
        # sess.run(tf.variables_initializer(vars_to_init + [n_iters_tf]))
        # sess.run(tf.global_variables_initializer())
        print('trainable:', tf.trainable_variables(), sep='\n')

    if hps.train:
        tb.run_summary(sess, train_writer, tb_merged, 0,
                       feed_dict={x_in:       x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to],
                                  flag_train: False})
        tb.run_summary(sess, test_writer, tb_merged, 0,
                       feed_dict={x_in:       x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to],
                                  flag_train: False})
        train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex],
                                                                          sess, hps.batch_size)
        test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess,
                                                                      hps.batch_size)
        print(
            'Epoch {:d}  test_dr {:.3f} {:.3f}  train_dr {:.3f} {:.3f}  test_mse {:.5f} {:.5f}  train_mse {:.5f} {:.5f}'.
            format(0, test_dr_pd, test_dr_sm, train_dr_pd, train_dr_sm, test_mse_pd, test_mse_sm, train_mse_pd,
                   train_mse_sm))
        tb.write_summary(test_writer, [test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm],
                         ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], 0)
        tb.write_summary(train_writer, [train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm],
                         ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], 0)

        global_iter = 0
        for epoch in range(1, hps.n_epochs + 1):
            for x_train_batch, y_train_batch in get_next_batch(x_train, y_train, hps.batch_size, shuffle=True):
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
            train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, hps.batch_size)
            test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, hps.batch_size)
            print('Epoch {:d}  test_dr {:.3f} {:.3f}  train_dr {:.3f} {:.3f}  test_mse {:.5f} {:.5f}  train_mse {:.5f} {:.5f}'.
                  format(epoch, test_dr_pd, test_dr_sm, train_dr_pd, train_dr_sm, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))
            tb.write_summary(test_writer, [test_mse_pd, test_mse_sm, test_dr_pd, test_dr_sm],
                             ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], epoch)
            tb.write_summary(train_writer, [train_mse_pd, train_mse_sm, train_dr_pd, train_dr_sm],
                             ['main/mse_pd', 'main/mse_sm', 'main/det_rate_pd', 'main/det_rate_sm'], epoch)
            # Save the model on each epoch after half of epochs are done
            if epoch > hps.n_epochs // 2:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, '{}/{}'.format(model_path, model_name), global_step=epoch)
    else:
        tb.run_summary(sess, train_writer, tb_merged, 0,
                       feed_dict={x_in:       x_train[img_tb_from:img_tb_to], y_in: y_train[img_tb_from:img_tb_to],
                                  flag_train: False})
        tb.run_summary(sess, test_writer, tb_merged, 0,
                       feed_dict={x_in:       x_test[img_tb_from:img_tb_to], y_in: y_test[img_tb_from:img_tb_to],
                                  flag_train: False})
        pred_coords_pd, pred_coords_sm = get_predictions(x_test, y_test, sess)
        scipy.io.savemat('matlab/predictions.mat', {'flic_pred_pd': pred_coords_pd,
                                                    'flic_pred_sm': pred_coords_sm})
    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))


