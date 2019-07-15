import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial


def unit_step_func_0_1(x):
    sign = x > 0.5
    return sign, sign


def unit_step_func_neg1_1(x):
    sign = (x > 0) * 2 - 1
    return sign, sign


def split_batches(batch_size, whole_data, shuffle=False):
    indices = range(len(whole_data))
    if shuffle:
        random.shuffle(list(indices))
    split_data = []
    batchs_amount = len(whole_data) // batch_size
    for i in range(batchs_amount):
        batch_xs = []
        batch_ys = []
        for j in range(batch_size):
            single_data = whole_data[indices[i*batch_size+j]]
            batch_xs.append(single_data[0])
            batch_ys.append(single_data[1])
        split_data.append([batch_xs, batch_ys])
    return split_data


def accuracy_0_1(pred, y):
    return unit_step_func_0_1(pred)[0] - y == 0


def accuracy_neg1_1(pred, y):
    return unit_step_func_neg1_1(pred)[0] - y == 0


def sigmoid(x):
    return np.array(1 / (np.exp(-x) + 1))


def sigmoid_training(x):
    sig = sigmoid(x)
    return sig, sig * (1 - sig)


def sigmoid_neg1_1(x):
    return 2 / (np.exp(-x) + 1) - 1


def sigmoid_neg1_1_training(x):
    sig = sigmoid_neg1_1(x)
    return sig, 2*sig*(1-sig)


def relu_training(x):
    condition = x > 0.
    return x*condition, condition


def relu(x):
    return x*(x > 0.)


def mse_err_cost(y_true, y_pred):
    return 2*(y_true - y_pred), (y_true - y_pred) ** 2


def mae_err_cost(y_true, y_pred):
    return (y_true - y_pred), ((y_true - y_pred) ** 2)**0.5


class Perceptron:
    def __init__(self, input_size, use_bias=False, training_activation_func=(lambda x: x, 1)):
        self.use_bias = use_bias
        self.training_activation_func = training_activation_func
        if use_bias:
            self.weights = 0.001 * np.random.randn(1, input_size + 1)
        else:
            self.weights = 0.001 * np.random.randn(1, input_size)

    def predict(self, inputs, activation_func=None):
        if self.use_bias:
            inputs = np.concatenate((inputs, np.array([1])))
        product = np.matmul(self.weights, inputs)
        if activation_func is not None:
            pred, der = activation_func(product)
            return pred, der

        return product

    # ____________________________________________________________________
    #                           GRADIENT DESCENT
    # ____________________________________________________________________

    # ______________________Parameters and constants______________________
    # P(w, b)           - perceptron represented as function
    # N                 - amount of values in single inputs array
    # x                 - inputs for single prediction indexed [0,N)
    # y_true            - true output for x
    # y_pred            - result of single prediction for x
    # w                 - weights for corresponding inputs indexed [0, N)
    # b                 - bias
    # a                 - learning rate
    # f                 - activation function
    #
    # ___________________________Functions_______________________________
    #
    # P(w, b) = f(dot_product(w, x) + b) = y_pred
    # err(y_pred, y_true) = y_pred - y_true         - error of prediction
    # c(y_pred, y_true) = err(y_pred, y_true)^2     - mse cost function
    #
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
    # P(w) = f(dot_product(w, x)) = y_pred
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
    # ∇P(w) = ∇(f(dot_product(w, x)))
    # ∇dot_product(w, x) = x
    # ∇P(w) = ∇f(w) * x
    # ∇c(w) = 2*x
    # w = w - a*err(w)*2*x
    #   --- because 'a' is a hyperparameter, equation can be simplified to ---
    # w = w - a*err(w)*x
    # err(w) is error for given weights; it can be calculated for single x or for batches of x's
    # _________________________Implementation_____________________________
    def train_gradient_descent(self, inputs, y_true, l_rate, cost_func=mse_err_cost):
        pred, activation_der = self.predict(inputs, self.training_activation_func)
        err, cost = cost_func(y_true, pred)
        if self.use_bias:
            inputs = np.concatenate((inputs, np.array([1])))
        self.weights = self.weights + l_rate * inputs * err * activation_der
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
            pred, activation_der = self.predict(inputs, self.training_activation_func)
            err, cost = cost_func(y_true, pred)
            if self.use_bias:
                inputs = np.concatenate((inputs, np.array([1])))
            dif = l_rate * inputs * err * activation_der
            batch_difs.append(dif)
        update = sum(batch_difs)/len(xs)
        self.weights = self.weights + update
        return np.mean(np.array(cost))


def main():
    perc = Perceptron(2, True, sigmoid_training)
    batch_size = 8
    train_data = [[[1, 0], [1]], [[0, 0], [0]], [[1, 1], [1]], [[0, 1], [1]],
                  [[1, 0], [1]], [[0, 0], [0]], [[1, 1], [1]], [[0, 1], [1]],
                  [[1, 0], [1]], [[0, 0], [0]], [[1, 1], [1]], [[0, 1], [1]]]
    # train_data = [[[1, -1], [-1]], [[-1, -1], [-1]], [[1, 1], [1]], [[-1, 1], [-1]]]
    err = float('inf')
    init_l_rate = 0.1
    max_err = 0.1
    epoch_max = 1000
    epoch = -1
    epochs = []
    errs = []
    accs = []
    # indecies = list(range(len(train_data)))

    l_rate = init_l_rate
    split = partial(split_batches, batch_size)
    while err > max_err and epoch < epoch_max - 1:
        epoch += 1
        err = 0
        acc = 0
        # random.shuffle(indecies)
        # for i in range(len(train_data)):
        #     x, y = train_data[indecies[i]]
        #     cost = perc.train_gradient_descent(np.array(x), np.array(y), l_rate, mse_err_cost)
        #     pred = perc.predict(x)
        #     acc += accuracy_0_1(pred, y)
        #     err += cost
        # err = err / len(train_data)
        # acc = acc / len(train_data)
        # epochs.append(epoch + 1)
        # errs.append(err)
        # accs.append(acc)
        batches = split(train_data, True)
        for batch in batches:
            cost = perc.train_stochastic_gradient_descent(batch[0], batch[1], l_rate, mse_err_cost)
            b_acc = 0
            for i in range(len(batch[0])):
                pred = perc.predict(batch[0][i])
                print(batch[0][i], pred, batch[1][i])
                b_acc += accuracy_0_1(pred, batch[1][i])[0]
            b_acc /= batch_size
            print(b_acc)
            acc += b_acc
            err += cost
        accs.append(acc/len(batches))
        errs.append(err/len(batches))
        epochs.append(epoch)

    plt.plot(epochs, errs, label='errs')
    print(accs)
    plt.plot(epochs, accs, label='accs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

# End of file
