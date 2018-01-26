import numpy as np
from scipy import signal
import pickle
"""
This file is to prepare the pairwise distribution for MRF with star model

"""
print(pickle.format_version)
y_test = np.load('y_test_flic.npy')
print(y_test.shape)

coefs = np.array([[1, 8, 28, 56, 70, 56, 28, 8, 1]], dtype=np.uint16) / 256
kernel = coefs.T @ coefs

joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose']

joint_dependece = {'lsho': ['nose', 'lelb'], 'lelb': ['lsho', 'lwri'], 'lwri': ['lelb'],
                   'rsho': ['nose', 'relb'], 'relb': ['rsho', 'rwri'], 'rwri': ['relb'],
                   'lhip': ['nose'], 'rhip': ['nose'], 'nose': ['lsho', 'rsho', 'lhip', 'rhip']}

dict = {'lsho': 0, 'lelb': 1, 'lwri': 2, 'rsho': 3, 'relb': 4, 'rwri': 5, 'lhip': 6,
        'lkne': 7, 'lank': 8, 'rhip': 9, 'rkne': 10, 'rank': 11, 'leye': 12, 'reye': 13,
        'lear': 14, 'rear': 15, 'nose': 16, 'msho': 17, 'mhip': 18, 'mear': 19, 'mtorso': 20,
        'mluarm': 21, 'mruarm': 22, 'mllarm': 23, 'mrlarm': 24, 'mluleg': 25, 'mruleg': 26,
        'mllleg': 27, 'mrlleg': 28}

def compute_pairwise_distribution(joint, cond_j):
    """
    This computes a single histogram for a pair of (joint, cond_joint), and applies gaussian smooth 
    :param joint: e.g. 'lsho'
    :param cond_j: e.g. 'nose'
    :return: 180 x 120 pairwise distribution
    """
    hp_height = y_test.shape[1]
    hp_width = y_test.shape[2]
    # TODO: Maybe we should add some small value to zeros to tackle the log0 in MRF
    pd = np.zeros([hp_height*2, hp_width*2])  # the pairwise distribution is twice the size of the heat map
    # print(pd.shape)
    for i in range(y_test.shape[0]):  # for every single image, we note the distance between the joint and cond_j
        img_j = np.reshape(y_test[i, :, :, joint_ids.index(joint)], (y_test.shape[1], y_test.shape[2]))
        img_cj = np.reshape(y_test[i, :, :, joint_ids.index(cond_j)], (y_test.shape[1], y_test.shape[2]))
        xj, yj = np.where(img_j == np.max(img_j))
        xcj, ycj = np.where(img_cj == np.max(img_cj))
        pd[hp_height+(xj-xcj), hp_width+(yj-ycj)] = pd[hp_height+(xj-xcj), hp_width+(yj-ycj)] + 1  # count for the histgram
    pd = pd / np.float32(np.sum(pd))
    pd = signal.convolve2d(pd, kernel, mode='same', boundary='fill', fillvalue=0)
    return pd

pairwise_distribution = {}
for joint in joint_ids:
    for cond_j in joint_ids:
        if cond_j not in joint_dependece[joint]:
            continue
        print(joint+'_'+cond_j)
        pairwise_distribution[joint+'_'+cond_j] = compute_pairwise_distribution(joint, cond_j)

# print(type(pairwise_distribution['rhip_nose']),pairwise_distribution['rhip_nose'].shape, np.sum(pairwise_distribution['rhip_nose']))
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.imshow(np.uint8(pairwise_distribution['lhip_nose'])*255)
# plt.figure(2)
# plt.imshow(np.uint8(pairwise_distribution['rhip_nose'])*255)
# plt.show()


with open('pairwise_distribution.pickle', 'wb') as handle:
    pickle.dump(pairwise_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
