import numpy as np


class Tensor(object):
    """
    For successor version, there's only `Tensor` except np.ndarray
    passing in the computational graph.
    """

    # np.ndarray
    data = None

    # tensor, Only retained if current tensor is leaf or
    # forced to be retained.
    grad = None

    # function
    grad_fn = None

    # bool, if false, no need to compute grad of current tensor.
    requires_grad = False

    # bool, default false
    is_leaf = False

    # bool, default false
    is_retained = False

    def __init__(self,
                 data=None,
                 requires_grad=False,
                 is_leaf=False,
                 is_retained=False):
        self.data = data
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.is_retained = is_retained

    def backward(self):
        if not self.grad_fn:
            return

        """
        if not self.requires_grad:
            return
        """

        grads = self.grad_fn.backward(self.grad)

        # Recursively call current tensor's grad_fn's
        # inputs' backward function.
        fn_inputs = self.grad_fn.inputs

        for idx, fn_input in enumerate(fn_inputs):
            if not fn_input.requires_grad:
                continue

            # Set grad, immediate save grad,
            # delete if not retained in the subclasses.
            fn_input.grad = grads[idx]

            # If leaf, stop.
            if fn_input.is_leaf:
                continue

            fn_input.backward()

            # Delete grad if necessary, if not retained grad, set None.
            if not fn_input.is_retained:
                fn_input.grad = None

    def is_empty(self):
        if not isinstance(self.data, np.ndarray):
            return True
        return False

    def shape(self):
        assert isinstance(self.data, np.ndarray)
        return self.data.shape

    def add(self, a):
        assert not a.is_empty()
        assert self.shape() == a.shape()
        self.data += a.data



