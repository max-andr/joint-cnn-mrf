import tensorflow as tf
import os


print(1)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print(2)

# with tf.device('/cpu:0'):
a = tf.constant(5)
b = tf.constant(3)
r = a + b


with tf.Session() as sess:
    print(sess.run(r))

