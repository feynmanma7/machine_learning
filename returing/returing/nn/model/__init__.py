from returing.nn.tensor import Tensor

from returing.nn.operation import Operation
from returing.nn.operation.loss import mse
from returing.nn.operation.activation import sigmoid, relu

#from returing.nn.optimizer import Optimizer
from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)


class Model(Operation):

    batch_size = None
    epochs = None
    stop_criterion = None
    early_stopping = None

    X = None
    Y = None

    activation = None
    loss_fn = None
    optimizer = None

    W = None
    b = None

    loss_tensor = None

    """
    loss_fn is loss function
    loss_val is the Value of loss function 
    """

    def __init__(self,
                 loss=None,
                 batch_size=1,
                 epochs=1,
                 stop_criterion=1e-3,
                 early_stopping=1000
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

    def compile(self, loss_fn='mse',
                activation='sigmoid',
                optimizer='sgd',
                stop_criterion=1e-3,
                early_stopping=1000):

        if loss_fn == 'mse':
            self.loss_fn = mse.MSELoss()
        else:
            self.loss_fn = loss_fn
        assert isinstance(loss_fn, Operation)

        if optimizer == 'sgd':
            self.optimizer = sgd.SGD()
        else:
            self.optimizer = optimizer
        assert isinstance(self.optimizer, Tensor)

        if activation == 'sigmoid':
            self.activation = sigmoid.Sigmoid()
        elif activation == 'relu':
            self.activation = relu.ReLU
        else:
            self.activation = activation
        # assert isinstance(self.activation, Operation)

        self.stop_criterion = stop_criterion
        self.early_stopping = early_stopping

    """
    def fit(self, X, Y):
        #This is a in-efficient implementation of fit,
        #which loads the whole data into memory at first.
        raise NotImplementedError
    """

    def fit(self, X, Y,
            batch_size=1,
            epochs=1,
            verbose=0
            ):

        assert isinstance(X, Tensor)
        assert isinstance(Y, Tensor)
        assert isinstance(self.loss_fn, Operation)
        #assert self.activation == None or isinstance(self.activation, Operation)
        assert isinstance(self.optimizer, Tensor)

        self.X = X
        self.Y = Y

        self.batch_size = batch_size
        self.epochs = epochs

        for epoch in range(self.epochs):
            # ===== Forward,
            # Compute the total graph, get loss(Tensor).
            self.forward()

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

            # ===== Backward
            # Compute the gradient(np.ndarray) of all of the weights.
            self.backward()

            # ===== Optimizer
            # Update parameters one-step
            self.optimizer.step()


    def fit_batch(self, X, Y):
        pass

    def forward(self, *args):
        pass

    def predict(self, X):
        self.X = X # ! Bad way to pass value!

        self.W = self.best_W
        self.b = self.best_b

        Y_pred = self.forward()
        return Y_pred

    def backward(self):
        self.loss_tensor.backward()