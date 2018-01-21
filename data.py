import numpy as np


def read_flic():
    # TODO: read data
    # TODO: prepare 90x60 heat maps with gaussian blobs on the joints' locations
    coefs = 1 / 256 * np.array([[1, 8, 28, 56, 70, 56, 28, 8, 1]])
    kernel = coefs.T @ coefs
    # Note: variance is 2 here, not 2.25
    pass


def get_dataset(name):
    if name == 'flic':
        x_train, y_train, x_test, y_test = read_flic()

    # standardization
    x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
    x_train_pixel_std = x_train.std(axis=0)  # per-pixel std
    x_train = (x_train - x_train_pixel_mean) / x_train_pixel_std
    x_test = (x_test - x_train_pixel_mean) / x_train_pixel_std
    return x_train, y_train, x_test, y_test

