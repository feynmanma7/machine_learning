import numpy as np
np.random.seed(20170430)


class Function(object):

    # tuple of input tensor
    inputs = None

    # tuple of output tensor
    outputs = None

    # save context specifically for backward, optional.
    saved_context = None

    def __init__(self):
        pass

    def __call__(self, *args):
        # Build graph on the fly.

        # Set Node's input Edges.
        self.inputs = args

        # Set requires_grad True,
        # if one of the input.requires_grad is True.
        requires_grad = False
        for input_tensor in self.inputs:
            if input_tensor.requires_grad:
                requires_grad = True
                break

        outputs = self.forward(args)

        # Set output Edges' creator Node.
        for output_tensor in outputs:
            output_tensor.grad_fn = self

            # Propagate the requires_grad flag.
            if requires_grad:
                output_tensor.requires_grad = True

        return outputs

    def forward(self, inputs):
        # Input: tuple of input tensor
        # Output: tuple of output tensor
        # Must return a tuple,
        # e.g. return X,
        raise NotImplementedError

    def backward(self, grads):
        # Input: tuple of numpy.ndarray, gradient w.r.t outputs,
        # Output: tuple of numpy.ndarray, gradient w.r.t inputs.
        # If return, must return a tuple, e.g.: return X_grad.
        pass


