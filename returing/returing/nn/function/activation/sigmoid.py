from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
from returing.nn.util.activation import simgoid


class Sigmoid(Function):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs):
        X, = inputs

        y_pred_data = simgoid(X.data)
        y_pred = Tensor(y_pred_data)

        self.saved_context = y_pred_data

        return y_pred,

    def backward(self, grads):
        y_pred_data = self.saved_context

        # f = sigmoid, f' = f(1-f)
        X_grad_data = y_pred_data * (1. - y_pred_data)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            if isinstance(y_pred_grad, Tensor):
                X_grad_data *= y_pred_grad.data

        X_grad = Tensor(X_grad_data)

        return X_grad,
