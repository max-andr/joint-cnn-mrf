import tensorflow as tf


def det_rate(heat_map_pred, heat_map_target, normalized_radius=10, joints='all'):
    """
    For a given pixel radius normalized by the torso height of each sample, we count the number of images in
    the test set for which the distance between predicted position and ground truth is less than this radius.
    It was introduced in [MODEC: Multimodal Decomposable Models for Human Pose Estimation]
    (https://homes.cs.washington.edu/~taskar/pubs/modec_cvpr13.pdf).

    heat_map_pred: tensor of size [n_images, height, width, n_joints]
    heat_map_target: tensor of size [n_images, height, width, n_joints]
    normalized_radius: pixel radius normalized by the torso height of each sample
    """
    def get_joints_coords(hm):
        hm_height, hm_width, n_joints = int(hm.shape[1]), int(hm.shape[2]), int(hm.shape[3])
        # we want to circumvent the fact that we can't take argmax over 2 axes
        hm = tf.reshape(hm, [-1, hm_height * hm_width, n_joints])
        coords_raw = tf.argmax(hm, axis=1)  # [n_images, n_joints]
        # Now we obtain real spatial coordinates for each image and for each joint
        coords_x = coords_raw // hm_width
        coords_y = coords_raw - coords_x * hm_width
        coords_xy = tf.stack([coords_x, coords_y], axis=1)
        return tf.cast(coords_xy, tf.float32)

    lhip_idx, rsho_idx = 6, 3
    pred_coords, true_coords = get_joints_coords(heat_map_pred), get_joints_coords(heat_map_target)

    torso_distance = tf.norm(true_coords[:, :, lhip_idx] - true_coords[:, :, rsho_idx], axis=1, keep_dims=True)  # [n_images]
    normalized_dist = tf.norm(pred_coords - true_coords, axis=1) * 100 / torso_distance  # [n_images, n_joints]
    if joints != 'all':
        norm_dist_list = []
        for joint in joints:
            norm_dist_list.append(normalized_dist[:, joint])
        normalized_dist = tf.stack(norm_dist_list, axis=1)
    detection_rate = 100*tf.reduce_mean(tf.cast(tf.less_equal(normalized_dist, normalized_radius), tf.float32))
    return detection_rate

