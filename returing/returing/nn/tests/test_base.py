from returing.nn.tensor.tensor import Tensor

from returing.nn.function.base import add_fn, \
    reshape, get_sub_tensor, transpose, concat, sum_fn, dot_fn

from returing.nn.function.loss.mse import MSELoss

import numpy as np
np.random.seed(20170430)


def test_add():
    a = Tensor(np.array([[1, 2, 3, 4]]), requires_grad=True, is_leaf=True)
    b = Tensor(np.array([[2, 3, 5, 8]]), requires_grad=True, is_leaf=True)

    c, = add_fn.Add()(a, b)
    print(c.data.shape)

    d = Tensor(np.random.randn(1, 4))

    loss, = MSELoss()(c, d)
    print(loss.data)
    loss.backward()
    print(a.grad.data)
    print(b.grad.data)

    """
    assert np.array_equal(c.data, np.array([[3, 5, 8, 12]]))

    d, = add.Add()(a, c)
    print(d.data)
    d.backward()
    print(a.grad.data)
    print(b.grad.data)
    """


def test_reshape():
    raw_shape = (2, 3, 4)
    target_shape = (2, 4, 3)

    a = Tensor(np.arange(24).reshape(raw_shape), requires_grad=True, is_leaf=True)
    b, = reshape.Reshape(target_shape=target_shape)(a)
    print(b.data)
    b.backward()
    print(a.grad.data)


def test_get_sub_tensor():

    # raw_shape = (2, 3, 4)
    coord_tuple = ((0, 1), (1, 2), (1, 3))
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b, = get_sub_tensor.GetSubTensor(coord_tuple=coord_tuple)(a)
    print(b.data)
    b.backward()
    print(a.grad.data)


def test_transpose():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    axes = (1, 2, 0)
    b, = transpose.Transpose(axes=axes)(a)
    print(b.data.shape)
    b.backward()
    print(a.grad.data.shape)


def test_concat():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    c = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)

    arr = tuple([a, b, c])
    d, = concat.Concat()(*arr)

    #print(d.data.shape)

    y = Tensor(np.random.randn(3, 2, 3, 4))
    loss, = MSELoss()(d, y)
    print(loss.data)

    loss.backward()

    print(a.grad.data)

    """
    e, = sum_fn.Sum()(d)
    print(e.data)
    e.backward()
    print(a.grad.data.shape)
    print(b.grad.data.shape)
    print(c.grad.data)
    """


def test_sum():
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b, = sum_fn.BatchSum()(a)
    c = Tensor(np.random.randn(2, ), requires_grad=True, is_leaf=True)
    d, = add_fn.Add()(b, c)
    e, = add_fn.Add()(d, c)
    e.backward()
    print(a.grad.data)
    print(c.grad.data)


def test_dot():
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True, is_leaf=True)
    c, = dot_fn.Dot()(a, b)
    print(c.data.shape)
    c.backward()
    print(a.grad.data)
    print(b.grad.data)


if __name__ == '__main__':
    #test_add()
    #test_reshape()
    #test_get_sub_tensor()
    #test_transpose()
    #test_concat()
    #test_sum()
    test_dot()

