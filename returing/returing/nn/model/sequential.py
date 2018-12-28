from . import Model
from returing.nn.tensor import Tensor
from returing.nn.operation import Operation
from returing.nn.operation.base import Matmul, Add
from returing.nn.utils.initialization import random_init_tensor

import numpy as np
np.random.seed(20170430)


class Sequential(Model):

    def __init__(self, *args):
        super(Sequential, self).__init__(args)

        if isinstance(args, Model) or \
                isinstance(args, tuple):
            self.model_list = args
        else:
            self.model_list = None

    def add_model(self, model):
        assert isinstance(model, Model)

        if self.model_list:
            self.model_list.append(model)
        else:
            self.model_list = [model]

    """
    def compile(self, loss_fn='mse',
                activation='sigmoid',
                optimizer='sgd',
                stop_criterion=1e-3,
                early_stopping=1000):
                pass
    """

    def fit(self, X, Y,
            batch_size=1,
            epochs=1,
            verbose=0
            ):

        assert isinstance(X, Tensor) or isinstance(X, list)
        assert isinstance(Y, Tensor) or isinstance(Y, list)

        self.batch_size = batch_size
        self.epochs = epochs

        for epoch in range(self.epochs):
            # ===== Forward
            Y_pred = X
            for model in self.model_list:
                Y_pred = model.forward(Y_pred)

            # ===== Backward
            # Compute Loss
            # Compute the gradient(np.ndarray) of all of the weights.
            self.loss_tensor = self.loss_fn(Y, Y_pred)
            self.loss_tensor.backward()

            # === Record: cur_iter, cur_loss, cur_params
            # === Update: best_iter, min_loss
            self.cur_epoch = epoch
            self.cur_loss_val = np.asscalar(self.loss_tensor.data)
            if self.cur_loss_val < self.min_loss_val:
                self.min_loss_val = self.cur_loss_val
                self.best_epoch = self.cur_epoch
                self.best_W = self.W
                self.best_b = self.b

            if verbose == 1:
                print("Epoch %d, cur_loss=%.4f, "
                      "best_epoch=%d, min_loss=%.4f" %
                      (epoch, self.cur_loss_val,
                       self.best_epoch, self.min_loss_val))

            # ===== Optimizer
            # Update parameters one-step
            self.optimizer.step(self.loss_tensor)


    def forward(self, *args):
        Y_pred = args
        for model in self.model_list:
            Y_pred = model(Y_pred)

        return Y_pred
