from returing.nn.module.module import Module
from returing.nn.tensor.tensor import Tensor
from returing.nn.tensor.parameter import Parameter
from returing.nn.util.initialization import random_init
from returing.nn.function.base import linear_func

import numpy as np


class Dense(Module):

    inputs = None
    outputs = None
    parameters = None

    def __init__(self,
                 n_in_features,
                 n_out_features,
                 initializer=None,
                 activation=None,
                 is_bias=True):
        super(Dense, self).__init__()

        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.is_bias = is_bias
        self.activaion = activation

        if not initializer:
            initializer = random_init

        # Initialize parameters
        self.W = Parameter(
            data=initializer((n_in_features, n_out_features)))

        if self.is_bias:
            self.bias = Parameter(
                data=initializer((n_out_features, )))

    def forward(self, inputs):
        """
        Input: X, tensor
            X: [n_samples, n_in_features]

        Output: y_pred, tensor
            y_pred: [n_samples, n_out_features]

        y_pred = Activation(Linear(X, W, bias))
        """
        X, = inputs

        y_pred = linear_func.LinearFunc(self.n_in_features,
                                        self.n_out_features,
                                        self.is_bias)(X, self.W, self.bias)

        #y_pred = self.activaion(y_pred)

        return y_pred

    """
    def backward(self, grads):

        y_pred_grad, W_grad, bias_grad = grads

        a_grad_shape, b_grad_shape = self.saved_context

        a_grad = np.ones(a_grad_shape)
        b_grad = np.ones(b_grad_shape)

        c_grad = grads
        if isinstance(c_grad, np.ndarray):
            a_grad *= c_grad
            b_grad *= c_grad

        return a_grad, b_grad
    """










