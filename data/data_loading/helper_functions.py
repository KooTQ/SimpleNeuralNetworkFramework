import numpy as np


def normalization(y_train, y_test, normalize):
    dividing_factor = 255. / (normalize[1] - normalize[0])
    move_factor = normalize[0]

    x_train = y_train / dividing_factor
    x_test = y_test / dividing_factor
    print(normalize)
    print(dividing_factor)
    print(move_factor)
    x_train = x_train - move_factor
    x_test = x_test - move_factor
    return x_train, x_test


def gaussian_noising(x_train, x_test, normalize, noise_scale):
    train_noise = np.random.normal(normalize[0], normalize[1], x_train.shape)
    test_noise = np.random.normal(normalize[0], normalize[1], x_test.shape)
    x_train = x_train + noise_scale * train_noise
    x_test = x_test + noise_scale * test_noise
    x_test = np.clip(x_test, normalize[0], normalize[1])
    x_train = np.clip(x_train, normalize[0], normalize[1])

    return x_train, x_test


# End of file
