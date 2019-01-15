from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
from returing.nn.util.activation import tanh


class Tanh(Function):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, inputs):
        X, = inputs

        y_pred_data = tanh(X.data)
        y_pred = Tensor(y_pred_data)

        #self.saved_context = X.data
        self.saved_context = y_pred_data

        return y_pred,

    def backward(self, grads):
        #X_data = self.saved_context
        y_pred_data = self.saved_context

        """
        # tanh
        f(x) = tanh
        f'(x) = 1 - tanh ** 2
        
        # sigmoid
        f(x) = tanh(x) = 2 * sigmoid(2x) - 1
        f'(x) = 4 * sigmoid(2x) * (1 - sigmoid(2x))
        """
        #sig_2x = simgoid(2 * X_data)
        #X_grad_data = 4 * sig_2x * (1. - sig_2x)

        X_grad_data = 1 - y_pred_data ** 2

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            if isinstance(y_pred_grad, Tensor):
                X_grad_data *= y_pred_grad.data

        X_grad = Tensor(X_grad_data)

        return X_grad,
