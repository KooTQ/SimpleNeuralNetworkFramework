import numpy as np


def unit_step_func_0_1(x):
    sign = x > 0.5
    return sign, sign


def unit_step_func_neg1_1(x):
    sign = (x > 0) * 2 - 1
    return sign, sign


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

# End of file
