from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from data.data_loading.load_mnist import load_mnist_2d_arr

input_width = 28
input_height = 28
input_depth = 1
learning_rate = 0.001
classes = 10

input_layer = Input(shape=(input_width, input_height, 1))
layer1 = Conv2D(16, (7, 7), activation='relu', padding='same')(input_layer)
layer2 = Conv2D(32, (5, 5), activation='relu', padding='same')(layer1)
max_pool_lay2 = MaxPool2D((2, 2))(layer2)
layer3 = Conv2D(32, (5, 5), activation='relu', padding='same')(max_pool_lay2)
layer4 = Conv2D(16, (3, 3), activation='relu', padding='same')(layer3)
max_pool_lay3 = MaxPool2D((2, 2))(layer4)
flat = Flatten()(max_pool_lay3)
layer5 = Dense(32, activation='relu')(flat)
output_layer = Dense(10, activation='softmax')(layer5)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = SGD(lr=learning_rate)

model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy', 'mse'])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_2d_arr(False)
    print(x_train.shape)
    print(y_train.shape)
    model.summary()
    model.fit(x_train, y_train, epochs=1)
    res = model.evaluate(x_test, y_test)
    print(res)

# End of file
