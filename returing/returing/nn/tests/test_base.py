from returing.nn.tensor.tensor import Tensor

from returing.nn.function.base import add_func, \
    reshape_func, get_sub_tensor_func, transpose_func, concat_func, batch_sum_func

import numpy as np
np.random.seed(20170430)


def test_add():
    a = Tensor(np.array([[1, 2, 3, 4]]), requires_grad=True, is_leaf=True)
    b = Tensor(np.array([[2, 3, 5, 8]]), requires_grad=True, is_leaf=True)

    c, = add_func.AddFunc()(a, b)
    print(c.data)
    assert np.array_equal(c.data, np.array([[3, 5, 8, 12]]))

    d, = add_func.AddFunc()(a, c)
    print(d.data)
    d.backward()
    print(a.grad.data)
    print(b.grad.data)


def test_reshape():
    raw_shape = (2, 3, 4)
    target_shape = (2, 4, 3)

    a = Tensor(np.arange(24).reshape(raw_shape), requires_grad=True, is_leaf=True)
    b, = reshape_func.ReshapeFunc(target_shape=target_shape)(a)
    print(b.data)
    b.backward()
    print(a.grad.data)


def test_get_sub_tensor():

    # raw_shape = (2, 3, 4)
    coord_tuple = ((0, 1), (1, 2), (1, 3))
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b, = get_sub_tensor_func.GetSubTensorFunc(coord_tuple=coord_tuple)(a)
    print(b.data)
    b.backward()
    print(a.grad.data)


def test_transpose():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    axes = (1, 2, 0)
    b, = transpose_func.TransposeFunc(axes=axes)(a)
    print(b.data.shape)
    b.backward()
    print(a.grad.data.shape)


def test_concat():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    b = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)
    c = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)

    arr = tuple([a, b, c])
    #arr = [a, b, c]
    #arr = a, b, c

    #d, = concat_func.ConcatFunc()(a, b, c)
    d, = concat_func.ConcatFunc()(*arr)
    print(d.data)
    d.backward()
    print(a.grad.data)


def test_sum():
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, is_leaf=True)

    b, = batch_sum_func.BatchSumFunc()(a)

    c = Tensor(np.random.randn(2, ), requires_grad=True, is_leaf=True)
    print(b.data.shape)
    #b.backward()
    #print(a.grad.data)

    d, = add_func.AddFunc()(b, c)
    e, = add_func.AddFunc()(d, c)
    e.backward()
    print(a.grad.data)
    #print(b.grad.data)
    print(c.grad.data)



if __name__ == '__main__':
    #test_add()
    #test_reshape()
    #test_get_sub_tensor()
    #test_transpose()
    #test_concat()
    test_sum()

