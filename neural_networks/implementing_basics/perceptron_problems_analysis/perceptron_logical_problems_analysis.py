from functools import partial

from neural_networks.implementing_basics.logical_datasets_generators import split_batches, data_generator, \
    and_label_func, or_label_func, xor_label_func

from neural_networks.implementing_basics.costs_activations import sigmoid_training as activation_training, \
    sigmoid as activation, \
    accuracy_0_1 as accuracy

from neural_networks.implementing_basics.perceptron import Perceptron

import numpy as np


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


class LogicalExperimentsPerceptron:
    def __init__(self, inputs_amount, learning_rate, label_func, training_activation_func,
                 true_false, epoch_stop, cost_stop, batch_size=None):
        self.inputs_amount = inputs_amount
        self.label_func = partial(label_func, true_false)
        self.training_activation_func = training_activation_func
        self.l_rate = learning_rate
        self.true_false = true_false
        self.epoch_stop = epoch_stop
        self.cost_stop = cost_stop
        self.batch_size = batch_size
        self.perceptron = None

    def train_batch(self, train_data, eval_data):
        train_cost = float('inf')
        epoch = -1
        train_costs = []
        batch_cost = 0
        while train_cost > self.cost_stop and epoch < self.epoch_stop - 1:
            epoch += 1
            batches = split_batches(self.batch_size, train_data, True)
            print(batches)
            for batch in batches:
                cost = self.perceptron.train_stochastic_gradient_descent(batch[0], batch[1], self.l_rate)
                batch_cost += cost
            train_costs.append(batch_cost/len(batches[0]))
            batch_cost = 0
        ev, acc = self.evaluation(eval_data)
        return epoch, ev, acc, train_costs

    def train_one_by_one(self, train_data, eval_data):
        epoch_cost = float('inf')
        epoch = -1
        train_costs = []
        while epoch_cost > self.cost_stop and epoch < self.epoch_stop - 1:
            epoch += 1
            epoch_cost = 0
            for x, y in train_data:
                x = np.array(x)
                y = np.array(y)
                train_cost = self.perceptron.train_gradient_descent(x, y, self.l_rate)
                epoch_cost += train_cost
            epoch_cost /= len(train_data)
            train_costs.append(epoch_cost)
        eval_, acc = self.evaluation(eval_data)
        return epoch, eval_, acc, train_costs

    def evaluation(self, eval_data):
        evals = []
        acc = 0.
        for x, y_true in eval_data:
            x = np.array(x)
            y_true = np.array(y_true)
            y_pred = self.perceptron.predict(x)
            ev = ((y_pred - y_true) ** 2) ** 0.5
            evals.append(ev)
            acc += accuracy(activation(y_pred), y_true)
        acc = acc / len(eval_data)
        return evals, acc

    def perform_experiments(self, use_bias, repeats):
        epochs = []
        evals = []
        accs = []
        train_costs = []
        if self.batch_size is not None:
            train_func = self.train_batch
        else:
            train_func = self.train_one_by_one

        for i in range(repeats):
            train_data = data_generator(self.true_false, self.inputs_amount, self.label_func, True)
            eval_data = data_generator(self.true_false, self.inputs_amount, self.label_func, False)
            self.perceptron = Perceptron(self.inputs_amount, use_bias, training_activation_func=activation_training)
            epoch, ev, acc, train_cost = train_func(train_data, eval_data)
            epochs.append(epoch + 1)
            evals.append(ev)
            accs.append(acc)
            train_costs.append(train_cost)
        return epochs, train_costs, evals, accs

    def problem_analysis(self, repeat=5):

        no_bias_results = self.perform_experiments(False, repeat)
        bias_results = self.perform_experiments(True, repeat)
        print('No bias:')
        print_results(no_bias_results)
        print('With bias:')
        print_results(bias_results)
        print('\n\n')


def main():
    cost_stop = 0.01
    learning_rate = 0.01
    epoch_stop = 1000
    repeat = 3
    inputs_amount = 2
    true_false = (1, 0)
    print('And logical function:')
    and_experiments = LogicalExperimentsPerceptron(inputs_amount, learning_rate, and_label_func,
                                                   activation_training, true_false, epoch_stop, cost_stop)
    and_experiments.problem_analysis(repeat)

    print('Or logical function:')
    or_experiments = LogicalExperimentsPerceptron(inputs_amount, learning_rate, or_label_func,
                                                  activation_training, true_false, epoch_stop, cost_stop)
    or_experiments.problem_analysis(repeat)

    print('Xor logical function:')
    xor_experiments = LogicalExperimentsPerceptron(inputs_amount, learning_rate, xor_label_func,
                                                   activation_training, true_false, epoch_stop, cost_stop)
    xor_experiments.problem_analysis(repeat)


if __name__ == '__main__':
    main()

# End of file
