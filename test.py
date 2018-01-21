import numpy as np
"""
This file is to test if the data.py prepare the data correctly

"""

y_test = np.load('y_test_flic.npy')
x_test = np.load('x_test_flic.npy')
print('x_test shape is', x_test.shape)
i = np.random.randint(0, high=x_test.shape[0])
# i = 655
print('Show the %dth image and the heat map for nose:' % i)
y_test = y_test.astype(np.float32)
y_test = y_test / 256

coords = np.zeros([2, 11])
img = x_test[i, :, :, :]
img = np.reshape(img, (x_test.shape[1], x_test.shape[2], x_test.shape[3]))

for joint in range(11):
    print(joint)
    hmap = y_test[i, :, :, joint]
    hmap = np.reshape(hmap, (y_test.shape[1], y_test.shape[2]))
    print(hmap.shape)
    x, y = np.where(hmap == np.max(hmap))
    print(x, y)
    coords[:, joint] = [x, y]
coords = coords * 8
print(coords)

import matplotlib.pyplot as plt
plt.figure(1)
plt.imshow(np.uint8(img))
plt.figure(2)
plt.imshow(np.uint8(hmap))
plt.show()

