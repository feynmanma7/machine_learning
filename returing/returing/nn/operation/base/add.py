from returing.nn.operation.operation import Operation
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Add(Operation):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, input_list):
        assert isinstance(input_list, tuple)
        assert len(input_list) == 2

        a = input_list[0]
        assert not a.is_empty()

        b = input_list[1]
        assert not b.is_empty()

        assert a.shape() == b.shape()

        c = Tensor()
        c.data = a.data + b.data
        c.grad_fn = self
        if a.requires_grad or b.requires_grad:
            c.requires_grad = True

        # Build computational graph.
        self.input_shape = a.shape()
        self.output_shape = a.shape()
        self.input_list = [a, b]

        return c

    def backward(self, grad_out):

        a_grad_data = np.ones(self.input_shape)
        b_grad_data = np.ones(self.input_shape)

        if isinstance(grad_out, Tensor):
            assert grad_out.shape() == self.output_shape

            # Element-wise multiply.
            a_grad_data *= grad_out.data
            b_grad_data *= grad_out.data

        a_grad = Tensor(data=a_grad_data)
        b_grad = Tensor(data=b_grad_data)
        return a_grad, b_grad






