from returing.nn.tensor.tensor import Tensor
from returing.nn.module.dense import Dense
import numpy as np


def test_dense():
    n_samples = 4
    n_in_features = 3
    n_out_features = 2
    X = Tensor(np.arange(n_samples * n_in_features).reshape(
        (n_samples, n_in_features)))

    dense_layer = Dense(n_in_features, n_out_features)
    y_pred, = dense_layer(X)
    print(y_pred.data)
    y_pred.backward()
    print(dense_layer.W.data)
    print(dense_layer.bias.data)


if __name__ == '__main__':
    test_dense()