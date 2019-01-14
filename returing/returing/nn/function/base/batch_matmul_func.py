from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class BatchMatMulFunc(Function):

    def __init__(self):
        super(BatchMatMulFunc, self).__init__()

    def forward(self, inputs):
        X, W = inputs

        """
        X: [n_samples, shape]
        W: [shape]
        """

        # Use `broadcast` of numpy.
        y_pred = Tensor(X.data * W.data)

        self.saved_context = X.requires_grad, W.requires_grad, \
                             X, W

        return y_pred,

    def backward(self, grads):
        X_requires_grad, W_requires_grad, X, W = self.saved_context

        X_grad_data = None
        W_grad_data = None

        if X_requires_grad:
            # W: [shape]
            # X: [n_samples, shape]
            X_grad_data = np.repeat(W.data, X.data.shape[0]).reshape(X.data.shape)

            if isinstance(grads, tuple):
                # y_pred_grad: [n_samples, shape]
                y_pred_grad, = grads
                X_grad_data *= y_pred_grad.data

        if W_requires_grad:
            # X: [n_samples, shape]
            # W: [shape]
            W_grad_data = np.sum(X.data, axis=0)

            if isinstance(grads, tuple):
                # y_pred_grad: [n_samples, shape], same with X
                y_pred_grad, = grads
                W_grad_data *= np.sum(y_pred_grad.data, axis=0)

        X_grad = Tensor(X_grad_data)
        W_grad = Tensor(W_grad_data)

        self.saved_context = None

        return X_grad, W_grad
