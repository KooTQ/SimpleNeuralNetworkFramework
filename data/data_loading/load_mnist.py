from keras.datasets import mnist
from keras.utils import to_categorical


def load_mnist_flat(normalize=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if normalize is not None:
        x_train, x_test = normalization(x_train, x_test, normalize)

    return (x_train, y_train), (x_test, y_test)


def load_mnist_2d_arr(channels_first=False, normalize=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if channels_first:
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
    else:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if normalize is not None:
        x_train, x_test = normalization(x_train, x_test, normalize)
        
    return (x_train, y_train), (x_test, y_test)


def normalization(x_train, x_test, normalize):
    if normalize == -1:
        x_train /= 122.5
        x_test /= 122.5

        x_train -= 1
        x_test -= 1

    elif normalize == 0:
        x_train /= 255
        x_test /= 255

    return x_train, x_test


# End of file
