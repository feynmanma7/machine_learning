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

        # Create return tensor.
        c = Tensor()
        c.data = a.data + b.data # 1. data
        c.grad_fn = self # 2. grad_fn
        if a.requires_grad or b.requires_grad: # 3. requires_grad
            c.requires_grad = True

        # Build computational graph.
        self.input_list = [a, b]  # 1.input_list, for recursion
        self.output_shape = a.shape()  # 3. output_shape_list
        self.input_shape = a.shape() # 3. input_shape_list, optional
        # 4. save_tensors for backward, optional

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






