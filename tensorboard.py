import tensorflow as tf

N_IMG_TO_SHOW = 50


def colorize(hm, joint_name):
    # joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose']
    zero_tensor = tf.zeros_like(hm)
    if joint_name == 'nose':
        return tf.concat([hm, zero_tensor, zero_tensor], axis=3)
    elif joint_name in ['lsho', 'rsho']:
        return tf.concat([zero_tensor, hm, zero_tensor], axis=3)
    elif joint_name in ['lelb', 'relb']:
        return tf.concat([zero_tensor, zero_tensor, hm], axis=3)
    elif joint_name in ['lwri', 'rwri']:
        return tf.concat([hm, hm, zero_tensor], axis=3)
    elif joint_name in ['lhip', 'rhip']:
        return tf.concat([hm, zero_tensor, hm], axis=3)


def get_var_by_name(var_name_to_find):
    return [v for v in tf.trainable_variables() if v.name == var_name_to_find][0]


def var_summary(var, name, baisc_name='pre_activ_'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(baisc_name+name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        pos_fract = tf.reduce_sum(tf.cast(tf.greater(var, 0), tf.float32)) / tf.cast(tf.size(var), tf.float32)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('std', stddev)
        tf.summary.scalar('n_pos', pos_fract)
        tf.summary.histogram('histogram', var)


def run_summary(sess, writer, tb_op, cur_iter, feed_dict):
    summary = sess.run(tb_op, feed_dict=feed_dict)
    writer.add_summary(summary, cur_iter)


def main_summaries(grads_vars, loss, det_rate):
    with tf.name_scope('main'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('det_rate', det_rate)
        # Add histograms for gradients (only for weights, not biases)
        for grad, var in grads_vars:
            if 'weights' in var.op.name:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                grad_l2_norm = tf.norm(tf.reshape(grad, [-1]))
                tf.summary.scalar(var.op.name + '/gradients', grad_l2_norm)
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        w_conv1_full_img = tf.transpose(get_var_by_name('conv1_fullres/weights:0'), [3, 0, 1, 2])  # displayed in color
        tf.summary.image('w_conv1', w_conv1_full_img, N_IMG_TO_SHOW)
        w_conv1_half_img = tf.transpose(get_var_by_name('conv1_halfres/weights:0'), [3, 0, 1, 2])  # displayed in color
        tf.summary.image('w_conv1', w_conv1_half_img, N_IMG_TO_SHOW)


def show_img_plus_hm(x, hm, joint_ids, in_height, in_width, hm_name):
    # Adjust brightness of the image and feature map, so that the most bright pixel equals 1
    contrast_x = 1 / tf.reduce_max(x, axis=[1, 2, 3], keep_dims=True)
    contrast_hm = 1 / tf.reduce_max(hm, axis=[1, 2], keep_dims=True)
    var_summary(x, 'input', hm_name)
    var_summary(hm, 'hm', hm_name)
    var_summary(contrast_x, 'contrast_x', hm_name)
    var_summary(contrast_hm, 'contrast_hm', hm_name)
    hm_large = contrast_hm * tf.image.resize_images(hm, [in_height, in_width])
    img_plus_all_joints = contrast_x * x  # init before adding joints' hms in the loop
    for joint_id, joint_name in enumerate(joint_ids):
        hm_cur_joint = colorize(hm_large[:, :, :, joint_id:joint_id + 1], joint_name)
        img_plus_joint = tf.minimum(x + hm_cur_joint, 1)
        tf.summary.image('hm_{}_{}'.format(hm_name, joint_name), img_plus_joint, N_IMG_TO_SHOW)
        img_plus_all_joints = tf.minimum(img_plus_all_joints + hm_cur_joint, 1)
    tf.summary.image('img_plus_all_joints_' + hm_name, img_plus_all_joints, N_IMG_TO_SHOW)

