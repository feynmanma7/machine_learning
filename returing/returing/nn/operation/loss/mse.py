from returing.nn.tensor import Tensor
from returing.nn.operation import Operation
import numpy as np
np.random.seed(20170430)


class MSELoss(Operation):

    A = None
    B = None

    def __init__(self, name=None):
        super(MSELoss, self).__init__()
        self.A = Tensor
        self.B = Tensor
        self.op_name = 'mse_loss'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0] # Y
        self.B = args[1] # Y_pred

        assert self.A.data.shape == self.B.data.shape

        # loss = .5 * ((Y_pred - Y) ** 2) / n_samples

        n_samples = self.A.data.shape[0]
        loss_value = 0.5 * (np.sum((self.B.data - self.A.data) ** 2))\
                     / n_samples
        C = Tensor(loss_value)
        C.name = self.name
        C.grad_fn = self

        # A = Y is the label, which is constant.
        self.A.requires_grad = False

        # B = Y_pred
        if self.B.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        self.B.parent = C
        C.left_child = self.A
        C.right_child = self.B

        return C

    def backward(self, C_grad=1):
        assert self.A.data.shape == self.B.data.shape

        # loss = .5 * (Y_pred - Y) ** 2
        # Y_pred_grad = Y_pred
        # grad_mat = self.B.data # Y_pred

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = self.B.data * C_grad
            else:
                self.B.grad += self.B.data * C_grad