from returing.nn.tensor import Tensor

from returing.nn.operation import Operation
#from returing.nn.operation.loss import Loss

#from returing.nn.optimizer import Optimizer
from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)


class Model(Operation):

    batch_size = None
    max_iter = None
    stop_criterion = None
    early_stopping = None

    X = None
    Y = None

    W = None
    b = None

    loss_fn = None
    optimizer = None

    """
    loss_fn is loss function
    loss_val is the Value of loss function 
    """

    def __init__(self,
                 loss=None,
                 batch_size=1,
                 max_iter=1,
                 stop_criterion=1e-3,
                 early_stopping=1000
                 ):

        super(Model, self).__init__()

        self.loss_fn = loss
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.stop_criterion = stop_criterion
        self.early_stopping = early_stopping

        self.cur_iter = -1
        self.cur_loss_val = np.Infinity
        self.best_iter = -1
        self.min_loss_val = np.Infinity

    def compile(self, loss_fn,
                optimizer='sgd',
                batch_size=1,
                max_iter=1,
                stop_criterion=1e-3,
                early_stopping=1000):
        self.loss_fn = loss_fn
        assert isinstance(loss_fn, Operation)

        if optimizer == 'sgd':
            self.optimizer = sgd.SGD()
        else:
            self.optimizer = optimizer
        assert isinstance(self.optimizer, Tensor)

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.stop_criterion = stop_criterion
        self.early_stopping = early_stopping

    def fit(self, X, Y):
        """
        This is a in-efficient implementation of fit,
        which loads the whole data into memory at first.
        """
        raise NotImplementedError

    def fit_batch(self, X, Y):
        pass

    def forward(self, *args):
        pass

    def predict(self, X):
        raise NotImplementedError