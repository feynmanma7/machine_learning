from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np
np.random.seed(20170430)


class Maximum(Function):
    # Element-wise maximum for two ndarrays.

    def __init__(self):
        super(Maximum, self).__init__()

    def forward(self, inputs):
        A, B = inputs
        Y_pred_data = np.maximum(A.data, B.data)
        Y_pred = Tensor(Y_pred_data)

        self.saved_context = A.data.shape, 1. * (A == Y_pred_data)

        return Y_pred,

    def backward(self, grads):
        Y_pred_grad, = grads

        X_data_shape, A_grad_data = self.saved_context
        B_grad_data = 1. - A_grad_data

        if isinstance(Y_pred_grad, Tensor):
            A_grad_data *= Y_pred_grad.data
            B_grad_data *= Y_pred_grad.data

        A_grad = Tensor(A_grad_data)
        B_grad = Tensor(B_grad_data)

        return A_grad, B_grad

