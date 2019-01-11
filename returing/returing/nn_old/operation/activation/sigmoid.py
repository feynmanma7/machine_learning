from returing.nn_old.operation import Operation
from returing.nn_old.tensor import Tensor
from returing.nn_old.utils import sigmoid, sigmoid_grad

import numpy as np
np.random.seed(20170430)


class Sigmoid(Operation):
    # Unary function
    A = None
    C = None

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self.A = Tensor
        self.op_name = 'sigmoid'
        self.name = name

    def forward(self, *args):

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        self.A = args[0]

        # Sigmoid: f(x) = sigmoid(x)
        C = Tensor(sigmoid(self.A.data))

        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        C.left_child = self.A

        self.C = C
        return C

    def backward(self, C_grad=1):
        if not self.A:
            return

        assert isinstance(self.A.data, np.ndarray)

        """
        !!! Attention
        f'(C) / f'(A) = sigmoid_grad(C)
        y = sigmoid(x)
        
        f'(y) / f'(x) = y * (1 - y) = sigmoid_grad(y)
        
        Not sigmoid_grad(x)
        """

        grad_mat = sigmoid_grad(self.C.data)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = grad_mat * C_grad
            else:
                self.A.grad += grad_mat * C_grad