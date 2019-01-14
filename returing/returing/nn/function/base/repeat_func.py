from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class RepeatFunc(Function):

    def __init__(self, repeat_times=None):
        super(RepeatFunc, self).__init__()

        self.repeat_times = repeat_times

    def forward(self, inputs):
        X,  = inputs

        y_pred_data = np.repeat(X.data, self.repeat_times)
        y_pred = Tensor(y_pred_data)

        self.saved_context = X.data.shape

        return y_pred,

    def backward(self, grads):
        X_grad_shape = self.saved_context

        # assert self.repeat_times > 0
        X_grad_data = np.ones(X_grad_shape) * 1. / self.repeat_times

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            X_grad_data *= \
                np.mean(y_pred_grad.data.reshape((self.repeat_times, -1), \
                                             axis=0))

        X_grad = Tensor(X_grad_data)

        #self.saved_context = None

        return X_grad,