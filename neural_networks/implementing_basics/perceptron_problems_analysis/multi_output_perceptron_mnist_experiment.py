from neural_networks.implementing_basics.multioutput_perceptron import MultioutputPerceptron
from neural_networks.implementing_basics.costs_activations import sigmoid, sigmoid_training, mse_err_cost
from data.data_loading.load_mnist import load_mnist_flat
import numpy as np


# Surprisingly gets over 91% accuracy
def main():
    epochs = 20
    input_size = 28*28*1
    output_size = 10
    train, test = load_mnist_flat((0, 1))
    perc = MultioutputPerceptron(input_size, output_size, use_bias=True, training_activation_func=sigmoid_training)
    learning_rate = 0.05

    for epoch in range(epochs):
        cost = 0
        for i in range(len(train[0])):
            cost += perc.train_gradient_descent(train[0][i], train[1][i], learning_rate, mse_err_cost)
        cost = np.sum(cost)
        print('Epoch: ' + str(epoch+1) + ' of ' + str(epochs))
        print('Training avg cost: ' + str(cost/len(train[0])))
        ev_cost = 0
        acc = 0
        for i in range(len(test[0])):
            ev = perc.predict(test[0][i], activation_func=sigmoid)
            acc += 1 if np.argmax(test[1][i]) == np.argmax(ev) else 0
            ev_cost += mse_err_cost(test[1][i], ev)[1]
        print('Evaluation accuracy: ' + str(acc/len(test[0])))
        print('Evaluation avg cost: ' + str(np.sum(ev_cost)/len(test[0])))
        print('\n')


if __name__ == '__main__':
    main()

# End of file
