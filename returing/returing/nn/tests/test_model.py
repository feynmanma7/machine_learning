from returing.nn.tensor import Tensor
from returing.nn.model import linear, dnn
from returing.nn.operation.loss import mse
from returing.nn.operation.activation import sigmoid, relu

from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)


def test_Dense():

    n_samples = 5
    n_input = 4
    n_output = 3

    X = Tensor(np.random.randn(n_samples, n_input),
               requires_grad=False,
               name='X')
    old_X = X.data

    Y = Tensor(np.random.randn(n_samples, n_output),
               name='Y')
    old_Y = Y.data

    model = dnn.Dense(n_input, n_output, bias=True)

    loss_fn = mse.MSELoss()
    optim = sgd.SGD(lr=1e-3)
    activation = relu.ReLU()
    model.compile(loss_fn=loss_fn, optimizer=optim, activation=activation)

    model.fit(X, Y, verbose=0, epochs=100)

    print('Dense best_epoch=%s, min_loss=%.4f' %
          (model.best_epoch, model.min_loss_val))


def test_Linear():

    n_samples = 5
    n_input = 4
    n_output = 3

    X = Tensor(np.random.randn(n_samples, n_input),
               requires_grad=False,
               name='X')
    old_X = X.data

    Y = Tensor(np.random.randn(n_samples, n_output),
               name='Y')
    old_Y = Y.data

    model = linear.Linear(n_input, n_output, bias=True)

    loss_fn = mse.MSELoss()
    optim = sgd.SGD(lr=1e-3)
    model.compile(loss_fn=loss_fn, optimizer=optim)
    model.fit(X, Y, verbose=0, epochs=100)

    print('Linear best_epoch=%s, min_loss=%.4f' %
          (model.best_epoch, model.min_loss_val))


if __name__ == '__main__':
    test_Linear()
    test_Linear()
    test_Linear()
    test_Dense()
    test_Dense()
    test_Dense()