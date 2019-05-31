from data.data_loading.load_mnist import load_mnist_flat, load_mnist_2d_arr
from neural_networks.models.direct_models.fully_connected_neural_network import get_model as fnn_model
from neural_networks.models.direct_models.convolutional_neural_network import get_model as cnn_model


def fnn():
    (x_train, y_train), (x_test, y_test) = load_mnist_flat()
    print(x_train.shape)
    print(y_train.shape)

    print(x_train.transpose().shape)
    print(y_train.shape)
    print(y_test)
    model = fnn_model()
    model.summary()
    model.fit(x_train, y_train, epochs=5)
    res = model.evaluate(x_test, y_test)
    print(res)


def cnn():
    (x_train, y_train), (x_test, y_test) = load_mnist_2d_arr()
    print(x_train.shape)
    print(y_train.shape)

    print(x_train.transpose().shape)
    print(y_train.shape)
    print(y_test)
    model = cnn_model()
    model.summary()
    model.fit(x_train, y_train, epochs=5)
    res = model.evaluate(x_test, y_test)
    print(res)


if __name__ == '__main__':
    fnn()
    cnn()


# End of file
