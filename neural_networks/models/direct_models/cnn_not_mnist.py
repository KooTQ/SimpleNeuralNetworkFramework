from neural_networks.models.direct_models.convolutional_neural_network import get_model
from data.data_loading.load_not_mnist import load_not_mnist_2d_arr


def main():
    (x_train, y_train), (x_test, y_test) = load_not_mnist_2d_arr(False)
    print(x_train.shape)
    print(y_train.shape)
    model = get_model()
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    res = model.evaluate(x_test, y_test)
    print(res)


if __name__ == '__main__':
    main()

# End of file
