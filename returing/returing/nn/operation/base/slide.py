from returing.nn.operation import Operation
from returing.nn.tensor import Tensor
from returing.nn.utils import safe_read_dict

import numpy as np
np.random.seed(20170430)


class Sliding2D(Operation):
    # Composite Unary Operation

    def __init__(self, **kwargs):
        super(Sliding2D, self).__init__()

        self.width_idx = safe_read_dict(kwargs, 'width_idx', 0)
        self.height_idx = safe_read_dict(kwargs, 'height_idx', 0)
        self.kernel_size = safe_read_dict(kwargs, 'kernel_size', 1)
        self.stride = safe_read_dict(kwargs, 'stride', 1)

    def forward(self, *args):
        """
        # Input
        X: [width, height] (Padded Size)
        width_idx,
        height_idx,
        kernel_size,
        stride

        # Output
        Y_pred: [K, K], kernel_size * kernel_size
        Y_pred = X[W_i*S:W_i*S+K, H_i*S:H_i*S+K]

        Y_pred = X [width_idx * stride : width_idx * stride + kernel_size,
                    height_idx * stride : height_idx * stride + kernel_size]

        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        X = args[0]

        self.X = X  # 1.Save input tensors for current operation

        assert isinstance(X.data, np.ndarray)

        Y_pred_data = self.X.data[self.width_idx * self.stride :
                             self.width_idx  * self.stride + self.kernel_size,
                      self.height_idx * self.stride :
                      self.height_idx * self.stride + self.kernel_size]

        Y_pred = Tensor(Y_pred_data)
        Y_pred.grad_fn = self # 3. Set grad_fn for current operation

        Y_pred.left_child = X # 4. Set parent-child relationships.
        X.parent = Y_pred

        return Y_pred # 2. Return new Tensor

    def backward(self, grad_out=None):
        """
        grad_out: [K, K], np.ndarray

        X: [width, height], np.ndarray
        Modify the gradient of X[W_i*S:W_i*S+K, H_i*S:H_i*S+K],
        not the entire X
        """
        assert isinstance(self.X, Tensor)
        assert isinstance(self.X.data, np.ndarray)

        # For sliding operation, the gradient is 1.
        # grad_mat = np.ones((self.kernel_size, self.kernel_size)) #[K, K]

        W_i = self.width_idx
        H_i = self.height_idx
        S = self.stride
        K = self.kernel_size

        if not isinstance(grad_out, np.ndarray):
            grad_out = np.ones((K, K))

        if self.X.requires_grad:
            if isinstance(self.X.grad, np.ndarray):
                # In numpy `*` is element-wise multiply
                self.X.grad[W_i*S:W_i*S+K, H_i*S:H_i*S+K]\
                    += self.X.grad[W_i*S:W_i*S+K, H_i*S:H_i*S+K] * grad_out
            else:
                self.X.grad = np.zeros(self.X.data.shape)
                self.X.grad[W_i*S:W_i*S+K, H_i*S:H_i*S+K] = grad_out