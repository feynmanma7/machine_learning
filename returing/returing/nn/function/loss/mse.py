from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class MSELoss(Function):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs):
        y_pred, y = inputs

        #assert y_pred.data.shape == y.data.shape
        #assert y_pred.data.shape[0] > 0

        # loss_data = 1/2 * (y_pred - y)^2 / n_samples
        loss_data = np.mean(.5 * (y_pred.data - y.data) ** 2)
        loss_tensor = Tensor(loss_data)

        self.saved_context = y_pred, y

        return loss_tensor,

    def backward(self, grads):
        y_pred, y = self.saved_context

        n_samples = y_pred.data.shape[0]
        y_pred_grad_data = (y_pred.data - y.data) / n_samples

        if isinstance(grads, tuple):
            loss_grad, = grads

            if isinstance(loss_grad, Tensor):
            # loss_grad.data is a scalar.
                y_pred_grad_data *= loss_grad.data

        y_pred_grad = Tensor(y_pred_grad_data)

        #self.saved_context = None

        return y_pred_grad,
