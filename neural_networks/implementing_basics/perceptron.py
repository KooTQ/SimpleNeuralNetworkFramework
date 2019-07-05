import numpy as np


def unit_step_func(x):
    return np.array(list(map((lambda i: 1 if i > 0 else -1), x)))


def main():
    and_perceptron = Perceptron(2, False, activation_func=unit_step_func)
    train_data = [(np.array([1, 1]), np.array([1])),
                  (np.array([-1, 1]), np.array([-1])),
                  (np.array([1, -1]), np.array([-1]))]
    test_data = (np.array([-1, -1]),  np.array([-1]))
    test_cost = float('inf')
    max_cost = 0.0001
    while test_cost > max_cost:
        for x, y in train_data:
            y_pred = and_perceptron.predict(test_data[0])
            test_err, test_cost = mse_err_cost(y_pred, test_data[1])
            print("Testing cost: " + str(test_cost))
            print("Testing pred: " + str(y_pred))
            train_cost = and_perceptron.train_gradient_descent(x, y, 0.1)
            print("Training cost: " + str(train_cost) + "\n\n")
    y_pred = and_perceptron.predict(test_data[0])
    test_err, test_cost = mse_err_cost(y_pred, test_data[1])
    print("Testing cost: " + str(test_cost))
    print("Testing pred: " + str(y_pred))


def mse_err_cost(y_pred, y_true):
    return (y_pred - y_true), (y_pred - y_true) ** 2


def mae_err_cost(y_pred, y_true):
    return (y_pred - y_true), ((y_pred - y_true) ** 2)**0.5


class Perceptron:
    def __init__(self, input_size, use_bias=False, activation_func=(lambda x: x)):
        self.use_bias = use_bias
        self.activation_func = activation_func
        if use_bias:
            self.weights = 0.01 * np.random.randn(1, input_size + 1)
        else:
            self.weights = 0.01 * np.random.randn(1, input_size)

    def predict(self, inputs):
        if self.use_bias:
            inputs = np.concatenate((inputs, np.array([1])))
        print(inputs)
        print(self.weights)
        product = np.matmul(self.weights, inputs)
        print(product)
        result = self.activation_func(product)

        return result

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
        y_pred = self.predict(inputs)
        err, cost = cost_func(y_pred, y_true)
        if self.use_bias:
            inputs = np.concatenate((inputs, np.array([1])))
        self.weights = self.weights - inputs * l_rate * err
        return cost


if __name__ == '__main__':
    main()

# End of file
