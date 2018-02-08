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


def mrf_fixed(heat_map, pairwise_energies):
    """
    from Learning Human Pose Estimation Features with Convolutional Networks
    :param heat_map: is produced by model as the unary distributions: batch_size x 90 x 60 x n_joints
    :param pairwise_energies: the priors for a pair of joints, it's calculated from the histogram and is fixed 1x180x120xn_pairs
    :return heat_map_hat: the result from equation 1 in the paper: batch_size x 90 x 60 x n_joints
    """

    def relu_pos(x, eps=0.00001):
        return tf.maximum(x, eps)

    delta = 10 ** -6  # for numerical stability
    heat_map_hat = []
    for joint_id, joint_name in enumerate(joint_names):
        hm = heat_map[:, :, :, joint_id:joint_id + 1]
        hm = batch_norm(hm)
        marginal_energy = tf.log(relu_pos(hm))  # heat_map: batch_size x 90 x 60 x 1
        for cond_joint in joint_dependence[joint_name]:
            cond_joint_id = np.where(joint_names == cond_joint)[0][0]
            prior = tf.nn.softplus(pairwise_energies[joint_name + '_' + cond_joint])
            likelihood = relu_pos(heat_map[:, :, :, cond_joint_id:cond_joint_id + 1])
            bias = tf.nn.softplus(pairwise_biases[joint_name + '_' + cond_joint])

            marginal_energy += tf.log(conv_mrf(prior, likelihood) + bias + delta)
        heat_map_hat.append(marginal_energy)
    return tf.stack(heat_map_hat, axis=3)[:, :, :, :, 0]


def mrf_trainable(heat_map):
    # TODO: produce a new heat map using MRF
    return heat_map


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
    return tf.get_variable('weights', initializer=initial)


def bias_variable(shape, init=0.0, name='biases'):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(init, shape=shape)
    return tf.get_variable(name, initial)


def conv_layer(x, size, stride, n_in, n_out, name, last_layer=False):
    with tf.name_scope(name):
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


def tower_loss(scope, logits, labels):
    """Calculate the total loss on a single tower.
      Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = mean_squared_error(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


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


gpus = list(range(8))
gpu_number, gpu_memory = '0', 0.6
debug = True
train, restore_model, best_model_name = True, False, '2018-01-29 15:24:39'
model_path = 'models_ex'
time_start = time.time()
cur_timestamp = str(datetime.now())[:-7]  # get rid of milliseconds
tb_folder = 'tb'
tb_train_iter = '{}/{}/train_iter'.format(tb_folder, cur_timestamp)
tb_train = '{}/{}/train'.format(tb_folder, cur_timestamp)
tb_test = '{}/{}/test'.format(tb_folder, cur_timestamp)
tb_log_iters = False
# img_tb_from, img_tb_to = 450, 465
img_tb_from, img_tb_to = 70, 85  # 50, 65
n_eval_ex = 500

n_joints = 9
n_pairs = 8
x_train, y_train, x_test, y_test = get_dataset()
n_train, in_height, in_width, n_colors = x_train.shape[0:4]
n_test, hm_height, hm_width = y_test.shape[0:3]
if debug:
    n_train, n_test = 4000, 100  # for debugging purposes we take only a small subset
    x_train, y_train, x_test, y_test = x_train[:n_train], y_train[:n_train], x_test[:n_test], y_test[:n_test]
# Main hyperparameters
n_epochs = 100
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
    pairwise_energies, pairwise_biases = {}, {}
    for joint in joint_names:
        for cond_joint in joint_dependence[joint]:
            joint_key = joint + '_' + cond_joint
            tensor = tf.convert_to_tensor(pairwise_distr_np[joint_key], dtype=tf.float32)
            pairwise_energy_jj = tf.reshape(tensor, [1, tensor.shape[0].value, tensor.shape[1].value, 1])
            pairwise_energies[joint_key] = tf.get_variable('energy_'+joint_key, initializer=pairwise_energy_jj)
            pairwise_biases[joint_key] = bias_variable([1, hm_height, hm_width, 1], 0.0001, 'bias_'+joint_key)
    y_in = tf.placeholder(tf.float32, [None, hm_height, hm_width, n_joints], name='heat_map')
    flag_train = tf.placeholder(tf.bool, name='is_training')

    n_iters_tf = tf.get_variable('n_iters', initializer=0, trainable=False)

    lr_tf = tf.train.piecewise_constant(n_iters_tf, lr_decay_n_updates, lr_decay_coefs)



    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(len(gpus)):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)



    # Data augmentation: we apply the same random transformations both to images and heat maps
    # x1, hm_target = tf.cond(flag_train, lambda: augmentation.augment_train(x_in, y_in), lambda: (x_in, y_in))
    x, hm_target = x_in, y_in

    # The whole heat map prediction model is here
    hm_pred_pd_logit = model(x, n_joints, debug)
    hm_pred_pd = spatial_softmax(hm_pred_pd_logit)

    hm_pred_sm_logit = mrf_fixed(hm_pred_pd, pairwise_energies)
    hm_pred_sm = spatial_softmax(hm_pred_sm_logit)

    with tf.name_scope('loss'):
        mse_pd = mean_squared_error(hm_pred_pd, hm_target)
        mse_sm = mean_squared_error(hm_pred_sm, hm_target)
        loss = mse_pd + mse_sm + lmbd * weight_decay(var_pattern='weights')



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
    for key in pairwise_energies:
        tf.summary.image('pairwise_potential_' + key, pairwise_energies[key], 30)
        tf.summary.image('pairwise_biases_' + key, pairwise_biases[key], 30)
        tb.var_summary(pairwise_energies[key], 'pairwise_energies_' + key)
        tb.var_summary(pairwise_biases[key], 'pairwise_biases_' + key)
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
        uninit_vars = []
        for joint in joint_names:
            for cond_joint in joint_dependence[joint]:
                uninit_vars.append(pairwise_energies[joint + '_' + cond_joint])
                uninit_vars.append(pairwise_biases[joint + '_' + cond_joint])
        saver.restore(sess, model_path + '/' + best_model_name)
        sess.run(tf.variables_initializer(uninit_vars))

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
            # TODO: eval on the whole train/test
            test_mse_pd, test_mse_sm = eval_error(x_test[:n_eval_ex], y_test[:n_eval_ex], sess, batch_size)
            train_mse_pd, train_mse_sm = eval_error(x_train[:n_eval_ex], y_train[:n_eval_ex], sess, batch_size)
            print('Epoch: {:d}  test_mse_pd: {:.5f}  test_mse_sm: {:.5f}  train_mse_pd: {:.5f}  train_mse_sm: {:.5f}'.
                  format(epoch, test_mse_pd, test_mse_sm, train_mse_pd, train_mse_sm))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver.save(sess, model_path + '/' + cur_timestamp)

    train_writer.close()
    test_writer.close()
    train_iters_writer.close()
print('Done in {:.2f} min\n\n'.format((time.time() - time_start) / 60))

# TODO: turn on the weight decay
# TODO: set up PGM part: first try the star model without trainable weight, and then try the trainable MRF

# TODO: show a principled plot of test MSE between PD and SM.
# TODO: if joint training is not successful, try auxiliary classifier?

# Things that are not super important
# TODO: data: handle multiple people by incorporating an extra "torso-joint heatmap" (page 6)
# TODO: set up the option to continue training (since we should do it in 3 stages according to the paper)
# TODO: local contrast normalization: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf and
# https://github.com/bowenbaker/metaqnn/blob/master/libs/input_modules/preprocessing.py
# TODO: Data Augm.: try zero padding and random crops!
# TODO: spatial dropout

# Do for the final submission
# TODO: a readme on how to run our code (1: download FLIC dataset, 2: data.py, 3: pariwise_distr.py, ...)

"""
To mention in the final report:
- we do it much faster using BN
- we use an auxiliary classifier (Inception style) to make both PD and SM perform well
This goes in line with motivation from the paper.
- thus we can train the model end-to-end from scratch and much faster (30 minutes instead of 60 hours)
but we should train on FLIC+...
- we provide understanding on what the model learns
- we fix the mistakes from the paper and fill the gaps (leaves mixed feeling, how could it be that they mixed up
the dimensions of the convolutions; if you really train model X, you just translate it to the description)

The paper leaves mixed feeling, especially because they didn't explain what their SM learn.
On Fig. 5 they showed "a didactic example", which they most probably drew by hand.
It would be very interesting to see the pairwise heatmaps obtained after backprop. We show them: ...
Thus, our contribution is not only in practical implementation of the paper, but also in understanding what the proposed 
model actually learns.

Interesting Q: how detection rate correlates with MSE?
"""
