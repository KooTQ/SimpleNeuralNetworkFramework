from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from data.data_loading.load_mnist import load_mnist_flat


def get_model(input_width=28, input_height=28, input_depth=1, learning_rate=0.001, classes=10):

    input_layer = Input(shape=(input_width*input_height*input_depth,))

    layer1 = Dense(32, activation='relu')(input_layer)
    layer2 = Dense(64, activation='relu')(layer1)
    layer3 = Dense(32, activation='relu')(layer2)
    output_layer = Dense(classes, activation='softmax')(layer3)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = SGD(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy', 'mse'])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_flat()
    print(x_train.shape)
    print(y_train.shape)

    print(x_train.transpose().shape)
    print(y_train.shape)
    print(y_test)
    model = get_model()
    model.summary()
    model.fit(x_train, y_train, epochs=5)
    res = model.evaluate(x_test, y_test)
    print(res)


if __name__ == '__main__':
    main()

# Results ~92% accuracy on testing set, unseen during training process

# End of file
