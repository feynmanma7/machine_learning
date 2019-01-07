import numpy as np
np.random.seed(20170430)


class Operation(object):

    input_list = None

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def forward(self, input_list):
        raise NotImplementedError

    def backward(self, grad_out_list):
        pass
