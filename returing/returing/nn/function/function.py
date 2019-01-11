import numpy as np
np.random.seed(20170430)


class Function(object):

    inputs = None
    outputs = None
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
        raise NotImplementedError

    def backward(self, grads):
        pass


