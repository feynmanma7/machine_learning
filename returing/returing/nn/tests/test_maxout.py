from returing.nn.tensor.tensor import Tensor
from returing.nn.module.maxout import Maxout
import numpy as np
np.random.seed(20170430)

def test_maxout():
    n_samples = 10
    input_dim = 3
    hidden_dim = 4
    n_kernel = 5
    X = Tensor(data=np.random.randn(n_samples, input_dim))
    layer = Maxout(n_kernel=n_kernel,
                input_dim=input_dim,
                hidden_dim=hidden_dim)
    Y, = layer(X)

    print(Y.data.shape)
    Y.backward()
    print(layer.W_list[0].data.shape)




if __name__ == '__main__':
    test_maxout()