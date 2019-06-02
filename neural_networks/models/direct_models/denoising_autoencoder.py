from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, Conv2DTranspose
from keras.optimizers import SGD
from data.data_loading.load_mnist import load_mnist_2d_arr
import numpy as np
import cv2


def get_model(input_width=28, input_height=28, input_depth=1, learning_rate=0.001, momentum=0.8):

    input_layer = Input(shape=(input_width, input_height, input_depth,))

    x = Conv2D(8, (7, 7), activation='relu', padding='same')(input_layer)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPool2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2DTranspose(8, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=input_layer, outputs=decoded)
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=0)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model


def main():
    (y_train, _), (y_test, _) = load_mnist_2d_arr(False)
    y_train = y_train / 255
    y_test = y_test / 255
    noise_scale = 0.5
    train_noise = np.random.normal(0, 1, y_train.shape)
    test_noise = np.random.normal(0, 1, y_test.shape)
    x_train = y_train + noise_scale * train_noise
    x_test = y_test + noise_scale * test_noise
    x_test = np.clip(x_test, .0, 1.0)
    x_train = np.clip(x_train, .0, 1.0)
    print(x_train.shape)
    model = get_model(learning_rate=0.1)
    model.summary()
    model.fit(x_train, y_train, epochs=1)
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

# Results ~96% accuracy on testing set, unseen during training process

# End of file
