from returing.nn_old.operation import Operation
from returing.nn_old.tensor import Tensor

import numpy as np
np.random.seed(20170430)


class ReLU(Operation):
    # Unary function
    A = None

    def __init__(self, name=None):
        super(ReLU, self).__init__()
        self.A = Tensor
        self.op_name = 'relu'
        self.name = name

    def forward(self, *args):

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        self.A = args[0]

        # ReLU: f(x) = max(0, x)
        # For numpy, relu(x) = x * (x > 0), relu_grad(x) = 1 * (x > 0)
        #C = Tensor(np.clip(self.A.data, a_min=0, a_max=np.Infinity))
        C = Tensor(self.A.data * (self.A.data > 0))

        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        C.left_child = self.A

        return C

    def backward(self, C_grad=1):
        if not self.A:
            return

        assert isinstance(self.A.data, np.ndarray)

        # ReLU_grad = 1 if x > 0 else 0
        grad_mat = 1 * (self.A.data > 0)
        # grad_mat = np.ones(self.A.data.shape)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = grad_mat * C_grad
            else:
                self.A.grad += grad_mat * C_grad