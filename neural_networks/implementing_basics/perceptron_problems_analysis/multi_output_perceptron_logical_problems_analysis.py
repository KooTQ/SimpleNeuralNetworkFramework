import itertools
from functools import partial

from neural_networks.implementing_basics.perceptron import Perceptron, split_batches, \
    sigmoid_training as activation_training, \
    sigmoid as activation, \
    accuracy_0_1 as accuracy
from itertools import product
import numpy as np

noise_factor = 0.015
train_aug = 5
t = 1
f = 0


def noise_func(items):
    return np.random.randn(*items.shape) * noise_factor + items


def and_label_func(inputs):
    return [np.array(inputs), t if all(map((lambda x: x == t), inputs)) else f]


def or_label_func(inputs):
    return [np.array(inputs), t if any(map((lambda x: x == t), inputs)) else f]


def xor_label_func(inputs):
    return [np.array(inputs), t if sum(map((lambda x: x == t), inputs)) == 1 else f]


def data_generator(variables_amount, label_func, use_noise=False):
    data = [[f, t]] * variables_amount
    data = list(map(label_func, product(*data)))
    if use_noise:
        data = list(map(lambda x: [noise_func(x[0]), x[1]], data))
    return data


def train_batch(batch_size, perceptron, learning_rate, cost_stop, epoch_stop, train_data, eval_data):
    train_cost = float('inf')
    epoch = -1
    train_costs = []
    batch_cost = 0
    while train_cost > cost_stop and epoch < epoch_stop - 1:
        epoch += 1
        batches = split_batches(batch_size, train_data, True)
        for batch in batches:
            cost = perceptron.train_stochastic_gradient_descent(batch[0], batch[1], learning_rate)
            batch_cost += cost
        train_costs.append(batch_cost/len(batches[0]))
        batch_cost = 0
    ev, acc = evaluation(perceptron, eval_data)
    return epoch, ev, acc, train_costs


def train_one_by_one(perceptron, learning_rate, cost_stop, epoch_stop, train_data, eval_data):
    epoch_cost = float('inf')
    epoch = -1
    train_costs = []
    while epoch_cost > cost_stop and epoch < epoch_stop - 1:
        epoch += 1
        epoch_cost = 0
        for x, y in train_data:
            x = np.array(x)
            y = np.array(y)
            train_cost = perceptron.train_gradient_descent(x, y, learning_rate)
            epoch_cost += train_cost
        epoch_cost /= len(train_data)
        train_costs.append(epoch_cost)
    eval_, acc = evaluation(perceptron, eval_data)
    return epoch, eval_, acc, train_costs


def evaluation(perceptron, eval_data):
    evals = []
    acc = 0.
    for x, y_true in eval_data:
        x = np.array(x)
        y_true = np.array(y_true)
        y_pred = perceptron.predict(x)
        ev = ((y_pred - y_true) ** 2) ** 0.5
        evals.append(ev)
        acc += accuracy(activation(y_pred), y_true)
    acc = acc / len(eval_data)
    return evals, acc


def train_many(train_func, label_func, input_amount, use_bias, learning_rate, cost_stop, epoch_stop, repeats=10):
    epochs = []
    evals = []
    accs = []
    train_costs = []
    train_data = []
    for i in range(train_aug):
        train_data.append(data_generator(input_amount, label_func, False))

    train_data = list(itertools.chain.from_iterable(train_data))
    for i in range(repeats):
        eval_data = data_generator(input_amount, label_func, False)
        perceptron = Perceptron(input_amount, use_bias, training_activation_func=activation_training)
        epoch, ev, acc, train_cost = train_func(perceptron, learning_rate,
                                                cost_stop, epoch_stop, train_data, eval_data)
        epochs.append(epoch + 1)
        evals.append(ev)
        accs.append(acc)
        train_costs.append(train_cost)
    return epochs, train_costs, evals, accs


def problem_analysis(inputs_amount, train_func, learning_rate, cost_stop, epoch_stop, repeat, label_func):
    no_bias_results = train_many(train_func, label_func,
                                 inputs_amount, False, learning_rate, cost_stop, epoch_stop, repeat)
    bias_results = train_many(train_func, label_func,
                              inputs_amount, True, learning_rate, cost_stop, epoch_stop, repeat)
    print_results(no_bias_results)
    print_results(bias_results)
    print('\n')


def print_results(results):
    avg_epoch = np.mean(np.array(results[0]))
    avg_train_cost = np.mean(np.array(results[1]))
    min_train_cost = (np.min(np.array(results[1])), np.argmin(np.array(results[1])))
    avg_eval_cost = np.mean(np.array(results[2]))
    min_eval_cost = (np.min(np.array(results[2])), np.argmin(np.array(results[2])))
    avg_acc = np.mean(np.array(results[3]))
    max_acc = (np.max(np.array(results[3])), np.argmax(np.array(results[3])))

    print("Avg epoch till stop: ", avg_epoch)
    print("Avg train cost ", avg_train_cost)
    print("Minimal train cost, position of minimal cost: ", min_train_cost)
    print("Avg evaluation cost: ", avg_eval_cost)
    print("Minimal eval cost, position of minimal cost: ", min_eval_cost)
    print("Avg accuracy: ", avg_acc)
    print("Maximal accuracy, position of max acc: ", max_acc)
    print("\n")


def main():
    cost_stop = 0.01
    learning_rate = 0.01
    epoch_stop = 1000
    repeat = 1
    batch_size = 5
    train_func = partial(train_batch, batch_size)
    inputs_amount = 2
    problem_analysis(inputs_amount, train_func, learning_rate, cost_stop, epoch_stop, repeat, and_label_func)
    problem_analysis(inputs_amount, train_func, learning_rate, cost_stop, epoch_stop, repeat, or_label_func)
    problem_analysis(inputs_amount, train_func, learning_rate, cost_stop, epoch_stop, repeat, xor_label_func)


if __name__ == '__main__':
    main()

# End of file
