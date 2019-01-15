import numpy as np
np.random.seed(20170430)


def simgoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return 2 * simgoid(2 * x) - 1