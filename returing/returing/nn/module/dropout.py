from returing.nn.module.module import Module
from returing.nn.tensor.tensor import Tensor
from returing.nn.function.function import Function
import numpy as np
np.random.seed(20170430)


"""
def benoulli(p):
    return 1 if np.random.random() > p else 0
"""

def benoulli(p, shape):
    return 1 * (np.random.random(shape) < p)


class Dropout(Function):
    def __init__(self,
                 dropout_ratio=0,
                 is_training_mode=False):
        super(Dropout, self).__init__()

        # Ratio to dropout, 1 - dropout is ratio to retain.
        self.dropout_ratio = dropout_ratio
        self.is_training_mode = is_training_mode

    def forward(self, inputs):
        X, = inputs

        Y_pred_data = X.data.copy()
        mask = None

        if self.is_training_mode:
            retain_ratio = 1 - self.dropout_ratio
            mask = benoulli(retain_ratio, X.data.shape)
            assert retain_ratio > 0
            Y_pred_data *= (mask / retain_ratio)

        self.saved_context = X.data.shape, mask

        Y_pred = Tensor(Y_pred_data)
        return Y_pred,

    def backward(self, grads):
        Y_pred_grad, = grads

        X_data_shape, mask = self.saved_context
        X_grad_data = np.ones(X_data_shape)

        if self.is_training_mode:
            X_grad_data *= mask

        if isinstance(Y_pred_grad, Tensor):
            Y_pred_grad_data = Y_pred_grad.data
            X_grad_data *= Y_pred_grad_data

        X_grad = Tensor(X_grad_data)

        return X_grad,



