from . import Model
from returing.nn.utils.initialization import random_init_tensor
from returing.nn.operation.base import Matmul, Add
import numpy as np
np.random.seed(20170430)


class Sequential(Model):

    def __init__(self,
                 n_input,
                 n_output,
                 is_bias=True,
                 *args, **kwargs):
        super(Sequential, self).__init__(args, kwargs)

        self.n_input = n_input
        self.n_output = n_output
        self.is_bias = is_bias

        # Initialization weights and bias (if necessary)
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

            Y_pred = Activation()(matmul(X, W) + b)

            W: [n_input, n_output]
            b: [1, n_output]
        """
        self.Y_pred = Matmul()(self.X, self.W)

        if self.is_bias:
            self.Y_pred = Add()(self.Y_pred, self.b)

        self.Y_pred = self.activation(self.Y_pred)
        self.loss_tensor = self.loss_fn(self.Y, self.Y_pred)

        self.optimizer.set_loss_tensor(self.loss_tensor)

        return self.Y_pred
