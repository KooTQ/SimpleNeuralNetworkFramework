import numpy as np
import matplotlib.pyplot as plt
from neural_networks.implementing_basics.costs_activations import mse_err_cost, sigmoid_training, accuracy_0_1


class MultioutputPerceptron:
    def __init__(self, input_size, output_size, use_bias=False, training_activation_func=(lambda x: x, 1)):
        self.use_bias = use_bias
        self.output_size = output_size
        self.input_size = input_size
        self.training_activation_func = training_activation_func
        if use_bias:
            self.weights = 0.001 * np.random.randn(output_size, input_size + 1)
        else:
            self.weights = 0.001 * np.random.randn(output_size, input_size)

    def predict(self, inputs, activation_func=None):
        if self.use_bias:
            inputs = np.concatenate((inputs, np.array([1])), axis=-1)
        product = np.matmul(self.weights, inputs)
        if activation_func is not None:
            return activation_func(product)

        return product

    # ____________________________________________________________________
    #                           GRADIENT DESCENT
    # ____________________________________________________________________

    # ______________________Parameters and constants______________________
    # P(w, b)              - perceptron represented as function
    # N                 - amount of values in single inputs array
    # x                 - inputs for single prediction indexed [0,N)
    # y_true            - true output for x
    # y_pred            - result of single prediction for x
    # w                 - weights for corresponding inputs indexed [0, N)
    # b                 - bias
    # a                 - learning rate
    #
    # ___________________________Functions_______________________________
    #
    # P(w, b) = dot_product(w, x) + b = y_pred
    # err(y_pred, y_true) = y_pred - y_true         - error of prediction
    # c(y_pred, y_true) = err(y_pred, y_true)^2     - mse cost function
    #
    # _________________________Goal and theory____________________________
    #
    #           goal is to minimize c(y_pred, y_true)
    # minimize cost is equivalent to find global minimum for cost via tweaking parameters (weights and bias)
    # perceptron is starting in arbitrary position in parameter space (weights and bias)
    #
    # direction of steepest ascent in function's result in respect for each parameter (eg. w0)
    #       is equivalent to partial derivative of same function over same parameter (eg. dc(w)/d(w0)  )
    #
    # direction of steepest ascent in function's results in respect to entire parameter space (eg. w)
    #       is equivalent to gradient of same function (eg. ∇c(w) )
    #
    # similarly direction of steepest descent in functions results is minus gradient of same function (eg. -∇c(w) )
    #
    # value of cost is proportional to "how far-off" are functions parameters to global minimum
    #       therefore to single correction step in gradients descent is multiplied by error value (eg. -err(w)*∇c(w) )
    # single step in gradients descent direction is multiplied by learning rate factor (eg. -a*err(w)*∇c(w) )
    #
    # for simplification we can assume that N-th index of w is bias and N-th index of inputs is always 1
    #
    # w[N] = b
    # x[N] = 1
    #     --- therefore perceptron represented as function changes to ---
    # P(w) = dot_product(w, x) = y_pred
    # err(w) = P(w), y_true
    # err(w) = y_pred, y_true
    # c(w) = err(w)^2
    # ____________________Mathematical representation of goals____________
    # w = w - a*err(w)∇c(w)
    # ________________________Formulae' transformations_____________________
    # ∇c(w) = ∇(err(y_pred, y_true))^2
    # ∇c(w) = 2*∇(err(y_pred, y_true))
    # ∇c(w) = 2*∇(y_pred - y_true)
    # ∇c(w) = 2*∇(P(w) - y_true)
    # ∇c(w) = 2*∇P(w) - 2*∇y_true
    # ∇y_true = 0
    # ∇P(w) = ∇dot_product(w, x)
    # ∇P(w) = x
    # ∇c(w) =
    # w = w - a*err(w)*2*x
    #   --- because 'a' is a hyperparameter, equation can be simplified to ---
    # w = w - a*err(w)*x
    # err(w) is error for given weights; it can be calculated for single x or for batches of x's
    # _________________________Implementation_____________________________
    def train_gradient_descent(self, inputs, y_true, l_rate, cost_func=mse_err_cost):
        y_pred, der = self.predict(inputs, self.training_activation_func)
        err, cost = cost_func(y_pred, y_true)
        err = err.reshape((1, self.output_size))
        if self.use_bias:
            ones = np.array([1])
            inputs = np.concatenate((inputs, ones), axis=-1)
            inputs = inputs.reshape((self.input_size + 1, 1))
        else:
            inputs = inputs.reshape((self.input_size, 1))

        dif = l_rate * inputs * err * der
        self.weights = self.weights - dif.T
        return cost

    # ____________________________________________________________________
    #                     STOCHASTIC GRADIENT DESCENT
    # ____________________________________________________________________
    def train_stochastic_gradient_descent(self, xs, ys, l_rate, cost_func=mse_err_cost):
        batch_difs = []
        cost = float('inf')
        for i in range(len(xs)):
            inputs = xs[i]
            y_true = ys[i]
            y_pred, der = self.predict(inputs, self.training_activation_func)
            err, cost = cost_func(y_pred, y_true)
            if self.use_bias:
                inputs = np.concatenate((inputs, np.array([1])))
            dif = l_rate * inputs * err * der
            batch_difs.append(dif)
        update = sum(batch_difs)/len(xs)
        self.weights = self.weights + update
        return np.mean(np.array(cost))


def main():
    input_size = 2
    output_size = 2
    and_perceptron = MultioutputPerceptron(input_size=input_size, output_size=output_size,
                                           training_activation_func=sigmoid_training,
                                           use_bias=True)
    train_data = [(np.array([1, 1]), np.array([1, 0])),
                  (np.array([0, 0]), np.array([0, 1])),
                  (np.array([0, 1]), np.array([0, 1])),
                  (np.array([1, 0]), np.array([0, 1]))]

    err = float('inf')
    init_l_rate = 0.02
    max_err = 0.01
    epoch_max = 1000
    epoch = -1
    epochs = []
    errs = []
    accs = []
    l_rate = init_l_rate
    while err > max_err and epoch < epoch_max:
        epoch += 1
        err = 0
        acc = 0
        for x, y in train_data:
            train_cost = and_perceptron.train_gradient_descent(x, y, l_rate)
            pred = and_perceptron.predict(x)
            acc += accuracy_0_1(pred, y)
            err += train_cost
        err = np.mean(err / len(train_data))
        acc = acc/len(train_data)
        acc = np.sum(acc)/output_size
        epochs.append(epoch + 1)
        errs.append(err)
        accs.append(acc)
    plt.plot(epochs, errs, label='errs')
    plt.plot(epochs, accs, label='accs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

# End of file
