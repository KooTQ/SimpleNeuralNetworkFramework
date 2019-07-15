import numpy as np


def sign(x):
    return 1 if x >= 0 else -1


def sigmoid(x):
    return 1/(1 + np.e**(-x))


class PercListsSigmoid:
    def __init__(self, inputs_amount, learning_rate=0.1, use_bias=False):
        self.weights = np.random.rand(inputs_amount + use_bias) - 0.5
        self.learning_rate = learning_rate

    def feed_forward(self, xs):
        sum = 0
        for i in range(len(xs)):
            sum += self.weights[i] * xs[i]
        if len(xs) < len(self.weights):
            sum += self.weights[len(xs)]
        res = sigmoid(sum)
        return res

    def train_epoch(self, inputs_with_labels):
        for input, label in inputs_with_labels:
            res = self.feed_forward(input)
            err = label - res
            for i in range(len(input)):
                x = input[i]
                self.weights[i] = self.weights[i] - err*self.learning_rate*x*sigmoid(x)*(1-sigmoid(x))
            if len(input) < len(self.weights):
                self.weights[len(input)] = self.weights[len(input)] - err*self.learning_rate*sigmoid(1)*(1-sigmoid(1))


class PercListsSign:
    def __init__(self, inputs_amount, learning_rate=0.1, use_bias=False):
        self.weights = np.random.rand(inputs_amount + use_bias) - 0.5
        self.learning_rate = learning_rate

    def feed_forward(self, xs):
        sum = 0
        for i in range(len(xs)):
            sum += self.weights[i] * xs[i]
        if len(xs) < len(self.weights):
            sum += self.weights[len(xs)]
        res = sign(sum)
        return res

    def train_epoch(self, inputs_with_labels):
        for input, label in inputs_with_labels:
            res = self.feed_forward(input)
            err = label - res
            for i in range(len(input)):
                self.weights[i] = self.weights[i] + err * self.learning_rate * input[i]
            if len(input) < len(self.weights):
                self.weights[len(input)] = self.weights[len(input)] + err*self.learning_rate


def main():
    train_data = [([1, 1], 1), ([-1, -1], -1), ([1, -1], -1), ([-1, 1], -1)]
    use_bias = True
    learning_rate = 0.1
    epochs_amount = 20
    perc = PercListsSign(2, learning_rate=learning_rate, use_bias=use_bias)
    for _ in range(epochs_amount):
        for xs, y in train_data:
            print(perc.feed_forward(xs), y)
        perc.train_epoch(train_data)
        print("\n\n")

    for xs, y in train_data:
        print(perc.feed_forward(xs), y)

    train_data = [([1, 1], 1), ([0, 0], 0), ([1, 0], 0), ([0, 1], 0)]
    perc = PercListsSigmoid(2, learning_rate=learning_rate, use_bias=use_bias)
    for _ in range(epochs_amount):
        for xs, y in train_data:
            print(perc.feed_forward(xs), y)
        perc.train_epoch(train_data)
        print("\n\n")

    for xs, y in train_data:
        print(perc.feed_forward(xs), y)


if __name__ == '__main__':
    main()

# End of file
