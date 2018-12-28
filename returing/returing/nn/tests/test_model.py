from returing.nn.tensor import Tensor
from returing.nn.model import linear
from returing.nn.operation.loss import mse

from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)


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
    model.compile(loss_fn=loss_fn, max_iter=45, optimizer=optim)

    model.fit(X, Y)

    print('\n', model.X.data, '\n')
    print(old_X, '\n\n\n')

    print(old_Y, '\n')
    Y_pred = model.predict(X)
    print(Y_pred.data, '\n\n\n')

    print(model.best_W.data, '\n')
    print(model.best_b.data)


if __name__ == '__main__':
    test_Linear()