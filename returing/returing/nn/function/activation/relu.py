from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor


class ReLU(Function):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        X, = inputs
        #y_pred = Tensor(np.clip(X.data, 0, np.inf))
        y_pred = Tensor(X.data * (X.data > 0))

        self.saved_context = X

        return y_pred,

    def backward(self, grads):
        X = self.saved_context
        X_grad_data = 1. * (X.data > 0)

        if isinstance(grads, tuple):
            y_pred_grad, = grads
            X_grad_data *= y_pred_grad.data
        X_grad = Tensor(X_grad_data)

        #self.saved_context = None

        return X_grad,
