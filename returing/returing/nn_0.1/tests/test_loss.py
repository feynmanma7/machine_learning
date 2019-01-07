from returing.nn_old.tensor import Tensor
from returing.nn_old.operation.loss import mse
import numpy as np
np.random.seed(20170430)


def test_mse_loss():

    n_samples = 5
    n_output = 4

    Y = Tensor(np.random.randn(n_samples, n_output),
               name='Y')
    Y_pred = Tensor(np.random.randn(n_samples, n_output),
                    requires_grad=True,
                    name='Y_pred')

    loss_ = mse.MSELoss('loss')(Y, Y_pred)

    Y.print()
    Y_pred.print()

    loss_.print()
    loss_.backward()

    print(Y_pred.grad)


if __name__ == '__main__':
    test_mse_loss()