import numpy as np
np.random.seed(20170430)


class Tensor(object):
    """
    For successor version, there's only `Tensor` except np.ndarray
    passing in the computational graph.
    """

    # np.ndarray
    data = None

    # tensor, same data shape with data if exists.
    grad = None

    # operation
    grad_fn = None

    # bool, if false, no need to compute grad of current tensor.
    requires_grad = None

    def __init__(self,
                 data=None,
                 requires_grad=None):
        self.data = data
        self.requires_grad = requires_grad

    def backward(self):
        if not self.grad_fn:
            return

        if not self.requires_grad:
            return

        grad_out_list = self.grad_fn.backward(self.grad)

        if not isinstance(self.grad_fn.input_list, list):
            return

        assert len(self.grad_fn.input_list) == len(grad_out_list)

        for idx in range(len(self.grad_fn.input_list)):
            input_tensor = self.grad_fn.input_list[idx]

            if not input_tensor.requires_grad:
                continue

            # Update grad of input_tensor
            if isinstance(input_tensor.grad, Tensor) and \
                    not input_tensor.grad.is_empty():
                # If has grad, add current grad to original grad.
                input_tensor.grad.add(grad_out_list[idx])
            else:
                # If no grad, create a new tensor.
                input_tensor.grad = grad_out_list[idx]

            input_tensor.backward()

    def is_empty(self):
        if not isinstance(self.data, np.ndarray):
            return True
        return False

    def shape(self):
        assert isinstance(self.data, np.ndarray)
        return self.data.shape

    def add(self, a):
        assert (a, Tensor)
        assert not a.is_empty()
        assert self.shape() == a.shape()
        self.data += a.data



