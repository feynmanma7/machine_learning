from returing.nn_old.tensor import Tensor
from returing.nn_old.model import linear, dnn
from returing.nn_old.model import sequential

import numpy as np
np.random.seed(20170430)


def test_sequential():
    n_samples = 5
    n_input = 4
    n_output = 3

    n_tmp = 10

    X = Tensor(np.random.randn(n_samples, n_input),
               requires_grad=False,
               name='X')

    Y = Tensor(np.random.randn(n_samples, n_output),
               requires_grad=False,
               name='Y')

    li = linear.Linear(n_input=n_input, n_output=n_output)(X)
    li_2 = linear.Linear(n_input=n_output, n_output=n_tmp)(li)
    Y_pred = linear.Linear(n_input=n_tmp, n_output=n_output)(li_2)

    #print(Y_pred.data)

    model = sequential.Sequential()
    linear_model = linear.Linear(n_input=n_input, n_output=n_output)
    linear_model_2 = linear.Linear(n_input=n_output, n_output=n_tmp)
    linear_model_3 = linear.Linear(n_input=n_tmp, n_output=n_output)

    model.add_model(linear_model)
    model.add_model(linear_model_2)
    model.add_model(linear_model_3)

    model.compile()
    model.fit(X, Y, epochs=100, verbose=0)

    print('Linear best_epoch=%s, min_loss=%.4f' %
          (model.best_epoch, model.min_loss_val))

    model = sequential.Sequential()
    dense_model = dnn.Dense(n_input=n_input, n_output=n_output,
                            activation='relu', lr=1e-2)
    dense_model_2 = dnn.Dense(n_input=n_output, n_output=n_tmp,
                              activation='relu', lr=1e-2)
    dense_model_3 = dnn.Dense(n_input=n_tmp, n_output=n_output,
                              activation='sigmoid')

    model.add_model(dense_model)
    model.add_model(dense_model_2)
    model.add_model(dense_model_3)

    model.compile()
    model.fit(X, Y, epochs=100, verbose=0)

    print('Dense best_epoch=%s, min_loss=%.4f' %
          (model.best_epoch, model.min_loss_val))

    model = sequential.Sequential(
        dnn.Dense(n_input=n_input, n_output=n_output,
                            activation='relu', lr=1e-2),
        dnn.Dense(n_input=n_output, n_output=n_tmp,
                              activation='relu', lr=1e-2),
        dnn.Dense(n_input=n_tmp, n_output=n_output,
                              activation='sigmoid'))

    #model.add_model(dense_model)
    #model.add_model(dense_model_2)
    #model.add_model(dense_model_3)

    model.compile()
    model.fit(X, Y, epochs=100, verbose=0)

    print('Dense best_epoch=%s, min_loss=%.4f' %
          (model.best_epoch, model.min_loss_val))


if __name__ == '__main__':
    test_sequential()