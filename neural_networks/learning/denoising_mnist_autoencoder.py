import cv2
from neural_networks.models.direct_models.denoising_autoencoder import get_model
from data.data_loading.load_mnist import load_mnist_2d_arr
from data.data_loading.helper_functions import gaussian_noising


def main():
    noise_scale = 0.5
    normalize = (0., 1.)
    learning_rate = 0.1
    epochs = 2
    (y_train, _), (y_test, _) = load_mnist_2d_arr(channels_first=False, normalize=normalize)
    x_train, x_test = gaussian_noising(y_train, y_test, normalize, noise_scale)

    print(x_train.shape)
    model = get_model(learning_rate=learning_rate)
    model.summary()
    model.fit(x_train, y_train, epochs=epochs)
    res = model.evaluate(x_test, y_test)
    print(res)
    res = model.predict(x_test)
    mnist_evalset_input_path = "C:\\cv_portfolio\\simple_nn_framework\\data\\datasets\\mnist_evalset_input\\"
    mnist_evalset_output_path = "C:\\cv_portfolio\\simple_nn_framework\\data\\datasets\\mnist_evalset_output\\"
    for i in range(res.shape[0]//10):
        cv2.imwrite(mnist_evalset_input_path + str(i) + ".png", x_test[i]*255)
        cv2.imwrite(mnist_evalset_output_path + str(i) + ".png", res[i]*255)


if __name__ == '__main__':
    main()


# End of file
