import random

from keras.utils import to_categorical
import os
import cv2

small_not_mnist_path = "C:\\cv_portfolio\\simple_nn_framework\\data\\datasets\\notMNIST_small\\"
labels_amount = 10


def load_not_mnist_flat(normalize=(0, 255)):
    labels = range(labels_amount)
    paths_per_label = list(map(lambda x: get_files(small_not_mnist_path+chr(ord('A')+x)), labels))
    result_x = []
    result_y = []
    for label in labels:
        for path in paths_per_label[label]:
            pic = im_read_flat(path, normalize)
            if pic is not None:
                result_x.append(pic)
                result_y.append(to_categorical(label, labels_amount))
    max_ind = len(result_x) * 4 // 5
    return [result_x[0:max_ind], result_y[0:max_ind]], [result_x[max_ind:], result_y[max_ind:]]


def im_read_flat(path, normalize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(path)
        return None
    img = img.reshape((img.shape[0]*img.shape[1]))
    # img = img / 255. * (normalize[1] - normalize[0]) + normalize[0]
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
