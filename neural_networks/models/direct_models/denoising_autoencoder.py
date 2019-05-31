from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D
from keras.optimizers import SGD
from data.data_loading.load_mnist import load_mnist_2d_arr
import numpy as np
import cv2


def get_model(input_width=28, input_height=28, input_depth=1, learning_rate=0.001):

    input_layer = Input(shape=(input_width, input_height, input_depth,))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPool2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=input_layer, outputs=decoded)
    optimizer = SGD(lr=learning_rate)

    model.compile(optimizer='adadelta', loss='mae', metrics=['mse'])
    return model


def main():
    (y_train, _), (y_test, _) = load_mnist_2d_arr(False)
    train_noise = np.random.normal(125.5, 50, y_train.shape)
    test_noise = np.random.normal(125.5, 50, y_test.shape)
    x_train = y_train + train_noise
    x_test = y_test + test_noise
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
        cv2.imwrite(mnist_evalset_input_path + str(i) + ".png", x_test[i])
        cv2.imwrite(mnist_evalset_output_path + str(i) + ".png", x_test[i])


if __name__ == '__main__':
    main()

# Results ~96% accuracy on testing set, unseen during training process

# End of file
