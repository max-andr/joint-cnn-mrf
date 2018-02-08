import tensorflow as tf


def detection_rate(heat_map_pred, heat_map_target, normalized_radius=10, joints='all'):
    """
    For a given pixel radius normalized by the torso height of each sample, we count the number of images in
    the test set for which the distance between predicted position and ground truth is less than this radius.
    It was introduced in [MODEC: Multimodal Decomposable Models for Human Pose Estimation]
    (https://homes.cs.washington.edu/~taskar/pubs/modec_cvpr13.pdf).

    heat_map_pred: tensor of size [n_images, height, width, n_joints]
    heat_map_target: tensor of size [n_images, height, width, n_joints]
    normalized_radius: pixel radius normalized by the torso height of each sample
    """
    def get_joints_coords_old(heat_map):
        max_per_hm = tf.reduce_max(heat_map, axis=[1, 2], keep_dims=True)  # [n_images, 1, 1, n_joints]
        # Result is: [n_images*n_joints, 4]: img_id, joint_id, coord_x, coord_y
        joints_coords_raw = tf.where(tf.equal(heat_map, max_per_hm))
        # TODO: for debugging: heat_map=tf.constant([[[[1,2,3],[3,4,5]], [[5,6,7],[7,8,9]]]])

        joints_coords_list = []
        for i in range(n_joints):
            if_joint_i = tf.equal(joints_coords_raw[:, 1], i)  # [n_images]
            coords_joint_i = tf.boolean_mask(joints_coords_raw, if_joint_i)[:, 2:]
            joints_coords_list.append(coords_joint_i)  # [n_images, 2]: coord_x, coord_y
        print(joints_coords_list)
        joints_coords = tf.stack(joints_coords_list, axis=2)
        return tf.cast(joints_coords, tf.float32)

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
    n_joints = heat_map_target.shape[3]
    pred_coords, true_coords = get_joints_coords(heat_map_pred), get_joints_coords(heat_map_target)

    torso_distance = tf.norm(pred_coords[:, :, lhip_idx] - true_coords[:, :, rsho_idx], axis=1, keep_dims=True)  # [n_images]
    normalized_dist = tf.norm(pred_coords - true_coords, axis=1) * 100 / torso_distance  # [n_images, n_joints]
    if joints != 'all':
        norm_dist_list = []
        for joint in joints:
            norm_dist_list.append(normalized_dist[:, joint])
        normalized_dist = tf.stack(norm_dist_list, axis=1)
    det_rate = tf.reduce_mean(tf.cast(tf.less_equal(normalized_dist, normalized_radius), tf.float32))
    return det_rate

