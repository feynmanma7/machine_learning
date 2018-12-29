from returing.nn.operation import Operation
from returing.nn.tensor import Tensor
from returing.nn.utils import safe_read_dict

import numpy as np
np.random.seed(20170430)


class Padding2D(Operation):
    # Atom Unary Operation

    def __init__(self, **kwargs):
        super(Padding2D, self).__init__()

        # Assume zero-padding in the default,
        # here padding is the number to pad.
        self.padding = safe_read_dict(kwargs, 'padding', 0)
        # === padding
        # 'valid': no padding
        # 'same': let output same length with input
        # 'causal': dialated ???

    def forward(self, *args):
        """
        # Input
        X: [width, height]
        Y_pred: [width + 2P, height + 2P]
        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        X = args[0]

        self.X = X  # 1.Save input tensors for current operation

        assert isinstance(X.data, np.ndarray)

        width, height = X.data.shape
        P = self.padding

        # !!! Do Zero Padding Here
        Y_pred_data = np.zeros((width + 2 * P,
                               height + 2 * P))

        # Copy X.data into Y, leave paddings in the around.
        Y_pred_data[P:-P, P:-P] = X.data

        Y_pred = Tensor(Y_pred_data)
        Y_pred.grad_fn = self # 3. Set grad_fn for current operation

        Y_pred.left_child = X # 4. Set parent-child relationships.
        X.parent = Y_pred

        return Y_pred # 2. Return new Tensor

    def backward(self, grad_out=None):
        """
        grad_out: [width+2P, height+2P], np.ndarray

        X.grad: [width, height], np.ndarray
        """
        assert isinstance(self.X, Tensor)
        assert isinstance(self.X.data, np.ndarray)

        # For padding operation, the gradient is 1.
        grad_mat = np.ones(self.X.data.shape) #[width, height]

        if not isinstance(grad_out, np.ndarray):
            x_grad_out = np.ones(self.X.data.shape)
        else:
            P = self.padding
            x_grad_out = grad_out[P:-P, P:-P]

        if self.X.requires_grad:
            if isinstance(self.X.grad, np.ndarray):
                # In numpy `*` is element-wise multiply
                self.X.grad += grad_mat * x_grad_out
            else:
                self.X.grad = grad_mat * x_grad_out