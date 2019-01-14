from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Padding2D(Function):

    def __init__(self, padding=None):
        super(Padding2D, self).__init__()
        self.padding = padding

    def forward(self, inputs):
        # X: [n_samples, n_in_ch, in_w, in_h]
        X, = inputs
        (n_samples, n_in_ch, in_w, in_h) = X.data.shape

        p = self.padding
        target_shape = (n_samples, n_in_ch, in_w+2*p, in_h+2*p)
        y_pred_data = np.zeros(target_shape)
        y_pred_data[:, :, p:p+in_w, p:p+in_h] = X.data

        y_pred = Tensor(y_pred_data)

        self.saved_context = X.data.shape, in_w, in_h

        return y_pred,

    def backward(self, grads):
        X_data_shape, in_w, in_h = self.saved_context
        self.saved_context = None

        X_grad_data = np.ones(X_data_shape)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            # y_pred_grad_data: [n_samples, n_in_ch, in_w+2*p, in_h+2*p]
            # X_grad_data: [n_samples, n_in_ch, in_w, in_h]
            p = self.padding
            X_grad_data *= y_pred_grad.data[:, :, p:p+in_w, p:p+in_h]

        X_grad = Tensor(X_grad_data)

        return X_grad,
