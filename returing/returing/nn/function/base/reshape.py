from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Reshape(Function):

    def __init__(self, target_shape=None):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, inputs):
        X, = inputs
        y_pred = Tensor(X.data.reshape(self.target_shape))

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_data_shape = self.saved_context
        X_grad_data = np.ones(X_data_shape)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            # y_pred_grad_data: target_shape
            # X_grad_data: raw_shape
            X_grad_data *= y_pred_grad.data.reshape(X_data_shape)
        X_grad = Tensor(X_grad_data)

        #self.saved_context = None

        return X_grad,
