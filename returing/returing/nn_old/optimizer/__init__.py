"""
from ..tensor import Tensor
import numpy as np
np.random.seed(20170430)


class Optimizer(Tensor):

    loss_ = None

    def __init__(self, loss_):
        super(Optimizer, self).__init__()

        # assert isinstance(loss_, Tensor)

        self.loss_ = loss_

    def step(self):
        raise NotImplementedError
"""