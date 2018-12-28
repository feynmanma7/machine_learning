from returing.nn.tensor import Tensor

from returing.nn.operation import Operation
from returing.nn.operation.loss import mse
from returing.nn.operation.activation import sigmoid, relu

from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)


class Model(Operation):

    batch_size = None
    epochs = None
    stop_criterion = None
    early_stopping = None

    activation = None
    loss_fn = None
    optimizer = None

    W = None
    b = None

    loss_tensor = None

    def __init__(self,
                 loss=None,
                 batch_size=1,
                 epochs=1,
                 stop_criterion=1e-3,
                 early_stopping=1000,
                 optimizer='sgd',
                 activation='relu',
                 loss_fn='mse'
                 ):

        super(Model, self).__init__()

        self.loss_fn = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.stop_criterion = stop_criterion
        self.early_stopping = early_stopping

        self.cur_epoch = -1
        self.cur_loss_val = np.Infinity
        self.best_epoch = -1
        self.min_loss_val = np.Infinity

        self.compile(loss_fn=loss_fn,
                     activation=activation,
                     optimizer=optimizer,
                     stop_criterion=stop_criterion,
                     early_stopping=early_stopping)

    def compile(self, loss_fn='mse',
                activation='sigmoid',
                optimizer='sgd',
                stop_criterion=1e-3,
                early_stopping=1000):

        if loss_fn == 'mse':
            self.loss_fn = mse.MSELoss()
        else:
            self.loss_fn = loss_fn
        assert isinstance(self.loss_fn, Operation)

        if optimizer == 'sgd':
            self.optimizer = sgd.SGD()
        else:
            self.optimizer = optimizer
        assert isinstance(self.optimizer, Tensor)

        if activation == 'sigmoid':
            self.activation = sigmoid.Sigmoid()
        elif activation == 'relu':
            self.activation = relu.ReLU()
        else:
            self.activation = activation
        assert isinstance(self.activation, Operation)

        self.stop_criterion = stop_criterion
        self.early_stopping = early_stopping

    def fit(self, X, Y,
            batch_size=1,
            epochs=1,
            verbose=0
            ):

        assert isinstance(X, Tensor) or isinstance(X, list)
        assert isinstance(Y, Tensor) or isinstance(Y, list)
        assert isinstance(self.loss_fn, Operation)
        #assert self.activation == None or isinstance(self.activation, Operation)
        assert isinstance(self.optimizer, Tensor)

        self.batch_size = batch_size
        self.epochs = epochs

        for epoch in range(self.epochs):
            # ===== Forward
            Y_pred = self.forward(X)

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


    def fit_batch(self, X, Y):
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, X):
        """
        Use Current Weights.
        """
        #self.W = self.best_W
        #self.b = self.best_b

        Y_pred = self.forward(X)
        return Y_pred

    """
    def backward(self):
        self.loss_tensor.backward()
    """