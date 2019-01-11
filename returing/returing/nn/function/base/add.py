from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Add(Function):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, inputs):

        a, b = inputs

        c_data = a.data + b.data
        c = Tensor(c_data)

        self.saved_context = a.data.shape, b.data.shape

        return c,

    def backward(self, grads):
        a_grad_shape, b_grad_shape = self.saved_context

        a_grad = np.ones(a_grad_shape)
        b_grad = np.ones(b_grad_shape)

        c_grad = grads
        if isinstance(c_grad, np.ndarray):
            a_grad *= c_grad
            b_grad *= c_grad

        return a_grad, b_grad










