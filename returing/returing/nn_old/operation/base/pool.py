from returing.nn_old.operation import Operation
from returing.nn_old.tensor import Tensor

from returing.nn_old.operation.base import Sum, ElementWiseMul
# from returing.nn_old.operation import

import numpy as np
np.random.seed(20170430)


class MaxPooling2D(Operation):

    def __init__(self):
        super(MaxPooling2D, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass


class AvgPooling2D(Operation):

    def __init__(self):
        super(AvgPooling2D, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass