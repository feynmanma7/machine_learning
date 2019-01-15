from returing.nn.function.activation.sigmoid import Sigmoid
from returing.nn.function.activation.tanh import Tanh
from returing.nn.tensor.tensor import Tensor
import numpy as np
np.random.seed(20170430)

def test_sigmoid():
    a = Tensor(np.random.randn(2, 3), requires_grad=True, is_leaf=True)
    b, = Sigmoid()(a)
    print(b.data)
    b.backward()
    print(a.grad.data)

def test_tanh():
    a = Tensor(np.random.randn(2, 3), requires_grad=True, is_leaf=True)
    b, = Tanh()(a)
    print(b.data)
    b.backward()
    print(a.grad.data)


if __name__ == '__main__':
    #test_sigmoid()
    test_tanh()