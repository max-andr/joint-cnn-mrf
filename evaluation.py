import tensorflow as tf


def detection_rate(heat_map_pred, heat_map_target, normalized_radius=10):
    """
    For a given pixel radius normalized by the torso height of each sample, we count the number of images in
    the test set for which the distance between predicted position and ground truth is less than this radius.
    It was introduced in [MODEC: Multimodal Decomposable Models for Human Pose Estimation]
    (https://homes.cs.washington.edu/~taskar/pubs/modec_cvpr13.pdf).

    heat_map_pred: tensor of size [n_images, height, width, n_joints]
    heat_map_target: tensor of size [n_images, height, width, n_joints]
    normalized_radius: pixel radius normalized by the torso height of each sample
    """
    def get_joints_coords(heat_map):
        max_per_hm = tf.reduce_max(heat_map, axis=[1, 2], keep_dims=True)  # [n_images, n_joints]
        # Result is: [n_images*n_joints, 4]: img_id, joint_id, coord_x, coord_y
        joints_coords = tf.where(tf.equal(heat_map, max_per_hm))
        # TODO: for debugging: heat_map=tf.constant([[[[1,2,3],[3,4,5]], [[5,6,7],[7,8,9]]]])
        return joints_coords

    lhip_idx, rsho_idx = 6, 3
    n_joints = heat_map_target.shape[3]
    pred_coords, true_coords = get_joints_coords(heat_map_pred), get_joints_coords(heat_map_target)
    bool_lhip, bool_rsho = tf.equal(true_coords[:, 1], lhip_idx), tf.equal(true_coords[:, 1], rsho_idx)  # [n_images]
    true_coords_lhip, true_coords_rsho = tf.boolean_mask(true_coords, bool_lhip), tf.boolean_mask(true_coords, bool_rsho)

    torso_distance = tf.norm(true_coords_lhip[:, 2:], true_coords_rsho[:, 2:], axis=[1, 2])  # [n_images * n_joints]
    actual_norm_distance = tf.norm(pred_coords[:, 2:] - true_coords[:, 2:], axis=1) * 100 / torso_distance  # [n_images * n_joints]
    det_rate = tf.reduce_mean(tf.less_equal(actual_norm_distance, normalized_radius))
    return det_rate

