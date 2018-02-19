from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import imageio
import skimage.transform

"""
This script generate the x_train, x_test, y_train, y_test for the further process from the FLIC dataset.

x_train, x_test is [n, 480, 720, 3], type = float32, pixels are from 0 to 1

y_train, y_test is [n, 60, 90, 10], type = float32

There're 3987 for training and 1016 for testing.
Note, that "torso-joint" is included as well. For the details why we need it, please read the original paper.
"""


def downsample_cube(myarr, factor, ignoredim=0):
    """
    Downsample a 3D array by averaging over *factor* pixels on the last two
    axes.
    """
    if ignoredim > 0:
        myarr = myarr.swapaxes(0, ignoredim)
    zs, ys, xs = myarr.shape
    crarr = myarr[:, :ys - (ys % int(factor)), :xs - (xs % int(factor))]
    dsarr = np.mean(np.concatenate([[crarr[:, i::factor, j::factor]
                                     for i in range(factor)]
                                    for j in range(factor)]), axis=0)
    if ignoredim > 0: dsarr = dsarr.swapaxes(0, ignoredim)
    return dsarr


def flip_backward_poses(flic_coords):
    """
    Flip left and right parts of backward facing people. It's used by Tompson et al. 2014 in their evaluation scripts.
    Looks like a cheating, but they claim that other people also use (used) this scheme.
    """
    hip_left, hip_right = dict['lhip'], dict['rhip']
    # if backward facing pose according to hips
    if flic_coords[:, hip_left][0] < flic_coords[:, hip_right][0]:
        for joint_left, joint_right in zip(['lwri', 'lelb', 'lhip', 'lsho'], ['rwri', 'relb', 'rhip', 'rsho']):
            joint_left, joint_right = dict[joint_left], dict[joint_right]
            coords_left_joint = flic_coords[:, joint_left]
            coords_right_joint = flic_coords[:, joint_right]
            flic_coords[:, joint_left] = coords_right_joint
            flic_coords[:, joint_right] = coords_left_joint
    return flic_coords


def how_many_backward_poses():
    """
    How many backward facing poses in the dataset.
    """
    # left_id, right_id = dict['lwri'], dict['rwri']
    left_id, right_id = dict['lhip'], dict['rhip']
    s_frontal = 0
    index = train_index
    for i in index:
        flic_coords = data_FLIC[i][2]
        # flic_coords = flip_backward_poses(flic_coords)
        coords_left = flic_coords[:, left_id] / 8
        coords_right = flic_coords[:, right_id] / 8
        s_frontal += coords_left[0] < coords_right[0]
        # print(coords_left[0], coords_right[0])
    print('frontal:', s_frontal, 'total:', len(index), 'fraction:', s_frontal / len(index))


def distances_hip_sho():
    """
    Show the distribution of torso heights.
    """
    index = train_index
    distances = []
    for i in index:
        flic_coords = data_FLIC[i][2]
        # flic_coords = flip_backward_poses(flic_coords)
        rhip = flic_coords[:, dict['rhip']]
        rsho = flic_coords[:, dict['rsho']]
        dist = np.sum((rhip - rsho) ** 2) ** 0.5
        distances.append(dist)
    distances = np.array(distances)
    print(np.min(distances), np.median(distances), np.max(distances))
    plt.hist(distances)


if __name__ == '__main__':
    # We tried a similar data preparation mentioned in "Learning human pose estimation features with convolutional
    # networks" (ICLR 2014) It unifies the scale on the training data, which seems as a good idea for the
    # scale-dependent spatial model.
    # However, it does not lead to improvements. So our recommendation is to set iclr_data_preparation = False.
    iclr_data_preparation = False
    data_FLIC = loadmat('data_FLIC.mat')
    data_FLIC = data_FLIC['examples'][0]
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose']  # , 'leye', 'reye',
    dict = {'lsho':   0, 'lelb': 1, 'lwri': 2, 'rsho': 3, 'relb': 4, 'rwri': 5, 'lhip': 6,
            'lkne':   7, 'lank': 8, 'rhip': 9, 'rkne': 10, 'rank': 11, 'leye': 12, 'reye': 13,
            'lear':   14, 'rear': 15, 'nose': 16, 'msho': 17, 'mhip': 18, 'mear': 19, 'mtorso': 20,
            'mluarm': 21, 'mruarm': 22, 'mllarm': 23, 'mrlarm': 24, 'mluleg': 25, 'mruleg': 26,
            'mllleg': 27, 'torso': 28}
    is_train = [data_FLIC[i][7][0, 0] for i in range(len(data_FLIC))]
    is_train = np.array(is_train)
    train_index = list(np.where(is_train == 1))[0]
    test_index = list(np.array(np.where(is_train == 0)))[0]
    print('# train indices:', len(train_index), '  # test indices:', len(test_index))
    # coefs = np.array([[1, 8, 28, 56, 70, 56, 28, 8, 1]], dtype=np.float32) / 256
    # coefs = np.array([[1, 4, 6, 4, 1]], dtype=np.float32) / 16
    coefs = np.array([[1, 2, 1]], dtype=np.float32) / 4  # maximizes performance
    # coefs = np.array([[1]])
    kernel = coefs.T @ coefs
    temp = round((len(kernel) - 1) / 2)
    pad = 5  # use padding to avoid the exceeding of the boundary

    # This part is for x_train and x_test
    orig_h, orig_w = 480, 720
    mode_height = 127.0  # roughly mode of the distribution of ||rhip - rsho||_2
    x_train, x_test, y_train_hmap, y_test_hmap = [], [], [], []
    for x, x_name, hmaps, hmaps_name, index in zip([x_train, x_test], ['x_train_flic', 'x_test_flic'], [y_train_hmap, y_test_hmap],
                                                   ['y_train_flic', 'y_test_flic'], [train_index, test_index]):
        for i in index:
            flic_coords = data_FLIC[i][2]
            flic_coords = flip_backward_poses(flic_coords)

            img = imageio.imread('./images_FLIC/' + data_FLIC[i][3][0])
            # img = downsample_cube(img, 2, ignoredim=2)  # the third dim
            img = img.astype(np.float32)
            img = img / 255

            # if training set
            if 'train' in x_name and iclr_data_preparation:
                # center of torso
                center = (flic_coords[:, dict['lsho']] + flic_coords[:, dict['rhip']] + flic_coords[:, dict['rsho']] +
                          flic_coords[:, dict['lhip']]) / 4
                center = (float(center[1]), float(center[0]))
                cur_height = np.sum((flic_coords[:, dict['rhip']] - flic_coords[:, dict['rsho']]) ** 2) ** 0.5
                scale = float(cur_height / mode_height)

                h1 = (1 - scale) / 2 * orig_h
                h2 = h1 + scale * orig_h
                w1 = (1 - scale) / 2 * orig_w
                w2 = w1 + scale * orig_w
                diff = [center[0] - orig_h / 2,
                        center[1] - orig_w / 2]  # diff between real center of human and center (480/2, 720/2)
                h1, h2 = round(h1 + diff[0]), round(h2 + diff[0])
                w1, w2 = round(w1 + diff[1]), round(w2 + diff[1])

                # but some coords h1,h2,w1,w2 can be negative => we need padding
                pad_h = max(0, -h1), max(h2 - orig_h, 0)
                pad_w = max(0, -w1), max(w2 - orig_w, 0)
                img_pad = np.pad(img, (pad_h, pad_w, (0, 0)), 'constant', constant_values=0)
                print('Before padding:', img.shape, '   after padding:', img_pad.shape)
                # changes are needed if we effectively changed our origin (after padding)
                h1, h2 = h1 + pad_h[0], h2 + pad_h[0]
                w1, w2 = w1 + pad_w[0], w2 + pad_w[0]
                img_crop = img_pad[h1:h2, w1:w2]
                img_final = skimage.transform.resize(img_crop, (orig_h, orig_w))
                x.append(img_final)
            else:
                x.append(img)

            hmap = []
            torso = (flic_coords[:, dict['lsho']] + flic_coords[:, dict['rhip']] + flic_coords[:, dict['rsho']] +
                     flic_coords[:, dict['lhip']]) / 4
            flic_coords[:, dict['torso']] = torso
            for joint in joint_ids + ['torso']:
                coords = np.copy(flic_coords[:, dict[joint]])
                # there are some annotation that are outside of the image (annotators did a great job!)
                coords[0], coords[1] = max(min(coords[1], orig_h), 0), max(min(coords[0], orig_w), 0)

                # Now we need y coordinates also to match
                if 'train' in x_name and iclr_data_preparation:
                    coords[0] = (coords[0] + pad_h[0] - h1) * img_final.shape[0] / img_crop.shape[0]
                    coords[1] = (coords[1] + pad_w[0] - w1) * img_final.shape[1] / img_crop.shape[1]

                coords /= 8
                heat_map = np.zeros([60, 90], dtype=np.float32)
                heat_map = np.lib.pad(heat_map, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
                coords = coords + pad
                h1_k, h2_k = int(coords[0] - temp), int(coords[0] + temp + 1)
                w1_k, w2_k = int(coords[1] - temp), int(coords[1] + temp + 1)
                heat_map[h1_k:h2_k, w1_k:w2_k] = kernel
                heat_map = heat_map[pad:pad + 60, pad:pad + 90]
                hmap.append(heat_map)
            hmap = np.stack(hmap, axis=2)
            hmaps.append(hmap)
        x = np.array(x, dtype=np.float32)
        np.save(x_name, x)
        print('Saved:', x_name, x.shape)

        hmaps = np.array(hmaps, dtype=np.float32)
        np.save(hmaps_name, hmaps)
        print('Saved:', hmaps_name, hmaps.shape)

