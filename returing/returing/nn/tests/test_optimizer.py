from returing.nn.tensor import Tensor

from returing.nn.operation.base import Add
from returing.nn.operation.loss import mse

from returing.nn.optimizer import sgd

import numpy as np
np.random.seed(20170430)

def test_sgd():

    n_samples = 5
    n_input = 4

    X_1 = Tensor(np.random.randn(n_samples, n_input),
               requires_grad=True,
               name='X_1')

    X_2 = Tensor(np.random.randn(n_samples, n_input),
                 requires_grad=True,
                 name='X_2')

    Y = Tensor(np.random.randn(n_samples, n_input),
           requires_grad=False,
           name='Y')

    Y_pred = Add()(X_1, X_2)
    loss_ = mse.MSELoss()(Y, Y_pred)
    loss_.backward()

    old_x1 = X_1.data
    old_x2 = X_2.data

    optim = sgd.SGD(loss_, lr=1e-1)
    optim.step()

    new_x1 = X_1.data
    new_x2 = X_2.data

    print("=" * 10)
    print(old_x1, '\n')
    print(new_x1, '\n')

    print("=" * 10)
    print(old_x2, '\n')
    print(new_x2, '\n')


    """
    X_1.print()
    X_2.print()
    Y.print()
    Y_pred.print()
    print(X_1.grad, '\n')
    print(X_2.grad, '\n')
    """



if __name__ == '__main__':
    test_sgd()