from . import Model
from returing.nn.tensor import Tensor
from returing.nn.utils.initialization import random_init_tensor

from returing.nn.operation.base import Matmul, Add

from returing.nn.operation import Operation

# from returing.nn.optimizer import Optimizer

import numpy as np
np.random.seed(20170430)


class Linear(Model):

    """
    X: [n_samples, n_input]
    Y_pred: [n_samples, n_output]

    Y_pred = matmul(X, W) + b

    W: [n_input, n_output]
    b: [1, n_output]
    """

    X = None
    Y = None
    Y_pred = None
    loss_fn = None
    loss_tensor = None

    W = None
    b = None

    best_W = None
    best_b = None

    n_input = None
    n_output = None

    def __init__(self,
                 n_input,
                 n_output,
                 is_bias=True,
                 *args, **kwargs):
        super(Linear, self).__init__(args, kwargs)

        self.n_input = n_input
        self.n_output = n_output
        self.is_bias = is_bias

        # Initialization weights and bias (if necessary)
        self.W = random_init_tensor((n_input, n_output),
                             requires_grad=True,
                             name='W')

        if self.is_bias:
            self.b = random_init_tensor((1, n_output),
                                 requires_grad=True,
                                 name='b')

    def fit(self, X, Y):

        assert isinstance(X, Tensor)
        assert isinstance(Y, Tensor)
        assert isinstance(self.loss_fn, Operation)
        assert isinstance(self.optimizer, Tensor)

        self.X = X
        self.Y = Y

        for iter in range(self.max_iter):
            # ===== Forward,
            # Compute the total graph, get loss(Tensor).
            self.forward()

            # === Record: cur_iter, cur_loss, cur_params
            # === Update: best_iter, min_loss
            self.cur_iter = iter
            self.cur_loss_val = np.asscalar(self.loss_tensor.data)
            if self.cur_loss_val < self.min_loss_val:
                self.min_loss_val = self.cur_loss_val
                self.best_iter = self.cur_iter
                self.best_W = self.W
                self.best_b = self.b

            print("Iter %d, cur_loss=%.4f, "
                  "best_iter=%d, min_loss=%.4f" %
                  (iter, self.cur_loss_val,
                   self.best_iter, self.min_loss_val))



            # ===== Backward
            # Compute the gradient(np.ndarray) of all of the weights.
            self.backward()


            # ===== Optimizer
            # Update parameters one-step
            self.optimizer.step()

    def predict(self, X):
        self.X = X # ! Bad way to pass value!

        self.W = self.best_W
        self.b = self.best_b

        Y_pred = self.forward()
        return Y_pred

    def forward(self, *args):

        """
        X: [n_samples, n_input]
        Y_pred: [n_samples, n_output]

        Y_pred = matmul(X, W) + b

        W: [n_input, n_output]
        b: [1, n_output]
        """

        self.Y_pred = Matmul()(self.X, self.W)

        if self.is_bias:
            self.Y_pred = Add()(self.Y_pred, self.b)

        # Return value of Loss_function(Operation) is Tensor
        self.loss_tensor = self.loss_fn(self.Y, self.Y_pred)

        # !!! Pass loss_val to optimizer, backward update parameters.
        self.optimizer.set_loss_tensor(self.loss_tensor)

        # self.cur_loss_tensor = self.loss_tensor # Tensor

        return self.Y_pred

    def backward(self):
        self.loss_tensor.backward()



