import numpy as np
np.random.seed(20170430)


class SGD(object):

    def __init__(self, lr=1e-3):
        # learning rate
        self.lr = lr

    def step(self, param_list):
        for parameter in param_list:
            parameter.data -= parameter.grad.data * self.lr