from returing.nn_old.tensor import Tensor
#from . import Optimizer
import numpy as np
np.random.seed(20170430)
np.seterr(invalid='ignore') # ??? Is this set ok ?

class SGD(Tensor):

    def __init__(self, lr=1e-3, *args, **kwargs):
        super(SGD, self).__init__(lr, *args, **kwargs)
        self.lr = lr

    def _update_parameters(self, node):

        if not node:
            return

        if not node.requires_grad:
            return

        if isinstance(node.grad, np.ndarray):
            node.data = node.data - self.lr * node.grad

        if node.left_child:
            self._update_parameters(node.left_child)
        if node.right_child:
            self._update_parameters(node.right_child)

    def step(self, loss_tensor):

        assert isinstance(loss_tensor, Tensor)

        # !!! Update parameters just one time
        self._update_parameters(loss_tensor)


