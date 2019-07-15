import random
import itertools
import numpy as np

noise_factor = 0.015


def noise_func(items):
    return np.random.randn(*items.shape) * noise_factor + items


def and_label_func(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if all(map((lambda x: x == t), inputs)) else f]


def or_label_func(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if any(map((lambda x: x == t), inputs)) else f]


def xor_label_func(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if sum(map((lambda x: x == t), inputs)) == 1 else f]


def and_label_func_multi_output(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if all(map((lambda x: x == 1), inputs)) else f]


def or_label_func_multi_output(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if any(map((lambda x: x == 1), inputs)) else f]


def xor_label_func_multi_output(true_false, inputs):
    t, f = true_false
    return [np.array(inputs), t if sum(map((lambda x: x == 1), inputs)) == 1 else f]


def data_generator(true_false, variables_amount, label_func, use_noise=False, aug_factor=1):
    t, f = true_false

    def inner():
        data = [[f, t]] * variables_amount
        data = list(map(label_func, itertools.product(*data)))
        if use_noise:
            data = list(map(lambda x: [noise_func(x[0]), x[1]], data))
        return data

    concat_data = []
    for i in range(aug_factor):
        concat_data.append(inner())
    concat_data = list(itertools.chain.from_iterable(concat_data))
    return concat_data


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

# End of file
