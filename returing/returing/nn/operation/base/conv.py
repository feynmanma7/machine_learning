from returing.nn.operation import Operation
from returing.nn.tensor import Tensor

import numpy as np
np.random.seed(20170430)


class Conv2D(Operation):
    """

    """

    A = None

    def __init__(self, name=None):
        super(Conv2D, self).__init__()
        self.A = Tensor
        self.op_name = 'conv2d'
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


    def backward(self, *args, **kwargs):
        pass