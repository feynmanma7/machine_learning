from returing.nn.tensor import Tensor
#from . import Optimizer
import numpy as np
np.random.seed(20170430)


class SGD(Tensor):

    lr = None
    loss_tensor = None

    def __init__(self, lr=1e-3,
        loss_val=None, **kwargs):
        super(SGD, self).__init__(loss_val, **kwargs)

        self.lr = lr
        self.loss_tensor = loss_val

    def set_loss_tensor(self, loss_tensor):
        self.loss_tensor = loss_tensor

    def _update_parameters(self, node):
        if not node:
            return

        # if node.requires_grad == True:
        if isinstance(node.grad, np.ndarray):
            node.data = node.data - self.lr * node.grad

        if node.left_child:
            self._update_parameters(node.left_child)
        if node.right_child:
            self._update_parameters(node.right_child)

    def step(self):
        # !!! Update parameters just one time

        if self.loss_tensor:
            self._update_parameters(self.loss_tensor)


