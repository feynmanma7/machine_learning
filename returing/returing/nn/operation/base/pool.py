from returing.nn.operation import Operation
from returing.nn.tensor import Tensor

from returing.nn.operation.base import Sum, ElementWiseMul
# from returing.nn.operation import

import numpy as np
np.random.seed(20170430)

class Pooling2D(Operation):

    def __init__(self):
        super(Pooling2D, self).__init__()

    def forward(self, *args):
        pass

    def backward(self, *args, **kwargs):
        pass