from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


def _to_slice_tuple(coord_tuple):
    # Input: coord_tuple, tuple of coordinate begin and end index
    # Output: slice_tuple, tuple of slice tuple
    # e.g.
    # --Input: ((0, 2), (0, 3), (0, 4), (1, 5)) ==>
    # --Output:(slice(0, 2), slice(0, 3), slice(0, 4), slice(1, 5))
    slice_tuple = []

    for begin, end in coord_tuple:
        slice_tuple.append(slice(begin, end))

    return tuple(slice_tuple)


class GetSubTensorFunc(Function):

    def __init__(self, coord_tuple=None):
        super(GetSubTensorFunc, self).__init__()

        # coordinate tuple must transform to slice tuple first.
        # e.g. ((0, 2), (0, 3), (0, 4), (1, 5)) ==>
        #      a[0:2, 0:3, 0:4, 1:5]
        self.coord_tuple = _to_slice_tuple(coord_tuple)

    def forward(self, inputs):
        # X: shape
        X, = inputs

        y_pred_data = X.data[self.coord_tuple]
        y_pred = Tensor(y_pred_data)

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_data_shape = self.saved_context

        assert len(X_data_shape) > 0

        X_grad_data = np.zeros(X_data_shape)
        X_grad_data[self.coord_tuple] = 1

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            # y_pred_grad_data: coord_shape
            # X_grad_data: raw_shape
            X_grad_data[self.coord_tuple] *= y_pred_grad.data

        X_grad = Tensor(X_grad_data)

        # `array` can be set to none to release memory, while variable not
        #self.saved_context = None

        return X_grad,
