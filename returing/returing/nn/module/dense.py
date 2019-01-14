from returing.nn.module.module import Module
from returing.nn.tensor.tensor import Tensor
from returing.nn.tensor.parameter import Parameter
from returing.nn.util.initialization import random_init
from returing.nn.function.base import linear

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
        self.activation = activation

        if not initializer:
            initializer = random_init

        # Initialize parameters
        self.W = Parameter(
            data=initializer((n_in_features, n_out_features)),
            name='W')
        self.parameters = [self.W]

        if self.is_bias:
            self.bias = Parameter(
                data=initializer((n_out_features, )),
                name='bias')
            self.parameters.append(self.bias)

    def forward(self, inputs):
        """
        Input: X, tensor
            X: [n_samples, n_in_features]

        Output: y_pred, tensor
            y_pred: [n_samples, n_out_features]

        y_pred = Activation(Linear(X, W, bias))
        """
        X, = inputs

        y_pred = linear.Linear(self.n_in_features,
                               self.n_out_features,
                               self.is_bias)(X, self.W, self.bias)

        y_pred = self.activation(y_pred)

        return y_pred











