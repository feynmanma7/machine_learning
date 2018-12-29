import numpy as np
np.random.seed(20170430)

def relu(x):
    return x * (x > 0)


def relu_grad(x):
    return 1 * (x > 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_grad(y):
    return y * (1 - y)


def safe_read_dict(dictory, key, default=0):
    return dictory[key] if key in dictory else default

