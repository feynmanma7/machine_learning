from returing.nn.operation import Operation
from returing.nn.tensor import Tensor
from returing.nn.utils import safe_read_dict

import numpy as np
np.random.seed(20170430)


class Padding2D(Operation):
    # Atom Unary Operation

    # TODO
    # Padding = 'same' or 'valid'

    def __init__(self, *args, **kwargs):
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
        X: [n_samples, width, height]

        # Output
        Y_pred: [n_samples, width + 2P, height + 2P]
        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        assert isinstance(args[0].data, np.ndarray)
        assert len(args[0].data.shape) == 3
        n_samples, width, height = args[0].data.shape
        self.n_samples = n_samples

        X = args[0]
        self.X = X  # 1.Save input tensors for current operation

        P = self.padding

        # !!! Do Zero Padding Here

        Y_pred_data = np.zeros((self.n_samples,
                                width + 2 * P,
                                height + 2 * P))

        # Copy X.data into Y, leave paddings in the around.
        if P == 0:
            Y_pred_data = X.data
        else:
            Y_pred_data[:, P:-P, P:-P] = X.data

        Y_pred = Tensor(Y_pred_data)
        Y_pred.grad_fn = self # 3. Set grad_fn & requires_grad for current operation
        if self.X.requires_grad:
            Y_pred.requires_grad = True

        Y_pred.left_child = X # 4. Set parent-child relationships.
        X.parent = Y_pred

        return Y_pred # 2. Return new Tensor

    def backward(self, padded_grad_out=None):
        """
        Note: If grad_out is not empty, it has the same shape with Y_pred.

        padded_grad_out: [n_samples, width+2P, height+2P], np.ndarray
        grad_out: [width, height]

        X.grad: [n_samples, width, height], np.ndarray
        """
        assert isinstance(self.X, Tensor)

        if not self.X.requires_grad:
            return

        assert isinstance(self.X.data, np.ndarray)

        # For padding operation, the gradient is 1.
        if not isinstance(padded_grad_out, np.ndarray):
            # X_data: [n_samples, width, height]
            # grad_out: [n_samples, width, height]
            grad_out = np.ones(self.X.data.shape)
        else:
            # assert padded_grad_out.shape == Y_pred.shape
            P = self.padding
            if P == 0:
                grad_out = padded_grad_out
            else:
                grad_out = padded_grad_out[:, P:-P, P:-P]

        # === !!! Note: Rely on the <b>Right</b>-align Broadcast of numpy.
        n_samples = self.X.data[0]

        # cur_grad (For Current Operation)
        # grad_out (Gradient from backwards), maybe empty.
        # If not total grad
        #   total_grad = cur_grad * grad_out
        # Else total_grad += cur_grad * grad_out

        cur_grad = np.ones(self.X.data.shape)

        # cur_grad: [n_samples, width, height]
        # grad_out: [n_samples, width, height]
        cur_grad = cur_grad * grad_out

        if isinstance(self.X.grad, np.ndarray):
            # In numpy `*` is element-wise multiply
            self.X.grad += cur_grad
        else:
            self.X.grad = cur_grad
