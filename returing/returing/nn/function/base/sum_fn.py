from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Sum(Function):

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs):
        X, = inputs
        y_pred = Tensor(np.sum(X.data))

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_data_shape = self.saved_context
        X_grad_data = np.ones(X_data_shape)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            X_grad_data *= y_pred_grad.data

        X_grad = Tensor(X_grad_data)

        return X_grad,



class BatchSum(Function):

    def __init__(self):
        super(BatchSum, self).__init__()

    def forward(self, inputs):
        # X: [n_samples, shape]
        X, = inputs

        # e.g: X.data.shape [n_samples, 1, 2, 3]
        # y_pred.data = np.sum(X.data.shape, axis=(1, 2, 3)),
        #   sum along `axis 0` (samples).
        y_pred = Tensor(np.sum(X.data,
                               axis=tuple(
                                   range(1, len(X.data.shape)))))

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_data_shape = self.saved_context

        X_grad_data = np.ones(X_data_shape)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            # X_grad_data: [n_samples, shape]
            # y_pred_grad: [n_samples, ]
            repeat_times = int(np.prod(X_data_shape) / np.prod(y_pred_grad.data.shape))
            X_grad_data *= np.repeat(y_pred_grad.data, repeat_times).reshape(X_data_shape)

        X_grad = Tensor(X_grad_data)

        #self.saved_context = None

        return X_grad,