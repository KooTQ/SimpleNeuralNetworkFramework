import random

from keras.utils import to_categorical
import numpy as np
import os
import cv2

small_not_mnist_path = "C:\\cv_portfolio\\simple_nn_framework\\data\\datasets\\notMNIST_small\\"
labels_amount = 10


def load_not_mnist_flat(normalize=(0, 255)):
    labels = range(labels_amount)
    paths_per_label = list(map(lambda x: get_files(small_not_mnist_path+chr(ord('A')+x)), labels))
    for label_set in paths_per_label:
        random.shuffle(label_set)
    result_x = []
    result_y = []
    for label in labels:
        for path in paths_per_label[label]:
            pic = im_read_flat(path, normalize)
            if pic is not None:
                result_x.append(pic)
                result_y.append(to_categorical(label, labels_amount))
    max_ind = len(result_x) * 4 // 5
    idx = np.random.permutation(len(result_x))
    result_x = np.array(result_x)[idx]
    result_y = np.array(result_y)[idx]
    train_x, train_y = result_x[0:max_ind], result_y[0:max_ind]

    test_x, test_y = result_x[max_ind:], result_y[max_ind:]
    return (train_x, train_y), (test_x, test_y)


def load_not_mnist_2d_arr(channels_first=False, normalize=(0, 255)):
    labels = range(labels_amount)
    paths_per_label = list(map(lambda x: get_files(small_not_mnist_path+chr(ord('A')+x)), labels))
    result_x = []
    result_y = []
    for label in labels:
        for path in paths_per_label[label]:
            pic = im_read_2d_arr(path, normalize, channels_first)
            if pic is not None:
                result_x.append(pic)
                result_y.append(to_categorical(label, labels_amount))
    max_ind = len(result_x) * 4 // 5
    idx = np.random.permutation(len(result_x))
    result_x = np.array(result_x)[idx]
    result_y = np.array(result_y)[idx]
    train_x, train_y = result_x[0:max_ind], result_y[0:max_ind]

    test_x, test_y = result_x[max_ind:], result_y[max_ind:]
    return (train_x, train_y), (test_x, test_y)


def im_read_flat(path, normalize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(path)
        return None
    img = img.reshape((img.shape[0]*img.shape[1]))
    img = img / 255. * (normalize[1] - normalize[0]) + normalize[0]
    return img


def im_read_2d_arr(path, normalize, channels_first):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(path)
        return None
    if channels_first:
        img = img.reshape((1, img.shape[0], img.shape[1]))
    else:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    img = img / 255. * (normalize[1] - normalize[0]) + normalize[0]
    return img


def get_files(path):
    result = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file:
                p = os.path.join(r, file)
                if '.png' in p:
                    result.append(p)
                else:
                    print('DAFAQ: ' + file)
    return result


if __name__ == '__main__':
    load_not_mnist_flat((0, 1))
# End of file
