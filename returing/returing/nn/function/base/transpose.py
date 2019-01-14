from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


def _get_transpose_back_axes(axes):
    """
    Input:
        axes: (0, 1, 2, 3) or (3, 0, 1, 2) like
    Output:
        new_axes, transpose back,
        for (0, 1, 2, 3) is (0, 1, 2, 3)
        for (3, 0, 1, 2) is (1, 2, 3, 0)
        which is the indexes of (0, 1, 2, 3).
    """
    return tuple(np.argsort(axes))


class Transpose(Function):

    """
    See numpy.transpose
    e.g.:
    a = np.random.randn(2, 3, 4), current axes = (0, 1, 2)
    a.tranpose(1, 0, 2)
    then a.shape = (3, 2, 4)
    """

    def __init__(self, axes=None):
        super(Transpose, self).__init__()

        self.axes = axes


    def forward(self, inputs):
        X, = inputs

        y_pred = Tensor(X.data.transpose(self.axes))

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_shape = self.saved_context

        X_grad_data = np.ones(X_shape)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            new_axes = _get_transpose_back_axes(self.axes)
            X_grad_data *= y_pred_grad.data.transpose(new_axes)

        X_grad = Tensor(X_grad_data)

        #self.saved_context = None

        return X_grad,
