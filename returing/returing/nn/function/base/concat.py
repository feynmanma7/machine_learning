from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Concat(Function):

    """
    Input: tuple of tensor, (shape, shape, ..., shape)
    Output: one tensor, (tuple_length, shape),
            (return tuple of tensor, but actually only `one` tensor),
            that concat data of  all of input tensors
    """

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs):

        y_pred_data = []

        for input_tensor in inputs:
            y_pred_data.append(input_tensor.data)

        input_tensor_shape = inputs[0].data.shape
        n_input = len(inputs)

        y_pred = Tensor(np.array(y_pred_data))

        #target_shape = [n_input] + list(input_tensor_shape)
        #y_pred_data = np.vstack(y_pred_data).reshape(target_shape)
        #y_pred = Tensor(y_pred_data)

        self.saved_context = input_tensor_shape, n_input

        return y_pred,

    def backward(self, grads):
        input_tensor_shape, n_input = self.saved_context

        inputs_grad = []
        for idx in range(n_input):
            input_tensor_grad_data = np.ones(input_tensor_shape)
            if isinstance(grads, tuple):
                y_pred_grad, = grads
                input_tensor_grad_data *= y_pred_grad.data[idx]

            inputs_grad.append(Tensor(input_tensor_grad_data))

        #self.saved_context = None

        return tuple(inputs_grad)
