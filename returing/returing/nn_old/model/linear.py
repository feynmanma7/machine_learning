from . import Model
from returing.nn_old.utils.initialization import random_init_tensor
from returing.nn_old.operation.base import MatMul, Add
from returing.nn_old.tensor import Tensor
import numpy as np
np.random.seed(20170430)


class Linear(Model):
    def __init__(self,
                 n_input,
                 n_output,
                 is_bias=True,
                 *args, **kwargs):
        super(Linear, self).__init__(args, kwargs)

        self.n_input = n_input
        self.n_output = n_output
        self.is_bias = is_bias

        # === Initialization weights and bias (if necessary)
        self.W = random_init_tensor((n_input, n_output),
                             requires_grad=True,
                             name='W')

        if self.is_bias:
            self.b = random_init_tensor((1, n_output),
                                 requires_grad=True,
                                 name='b')

    def forward(self, *args):
        """
        X: [n_samples, n_input]
        Y_pred: [n_samples, n_output]

        Y_pred = matmul(X, W) + b

        W: [n_input, n_output]
        b: [1, n_output]
        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        X = args[0]

        Y_pred = MatMul()(X, self.W)

        if self.is_bias:
            Y_pred = Add()(Y_pred, self.b)

        return Y_pred