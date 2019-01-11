from returing.nn.tensor.tensor import Tensor

from returing.nn.function.base import add

import numpy as np
np.random.seed(20170430)


def test_add():
    a = Tensor(np.array([[1, 2, 3, 4]]), requires_grad=True, is_leaf=True)
    b = Tensor(np.array([[2, 3, 5, 8]]), requires_grad=True, is_leaf=True)

    c, = add.Add()(a, b)
    print(c.data)
    assert np.array_equal(c.data, np.array([[3, 5, 8, 12]]))

    d, = add.Add()(a, c)
    print(d.data)
    d.backward()
    print(a.grad.data)
    print(b.grad.data)


if __name__ == '__main__':
    test_add()

