import numpy as np
np.random.seed(20170430)

from returing.nn_old.operation.base import *
from returing.nn_old.operation.activation import relu, sigmoid
from returing.nn_old.utils import *


# ==================TEST=======================
def test_Add_Subtract():
    a = Tensor(np.array([1, 2, 3, 4]), requires_grad=True, name='a')
    b = Tensor(np.array([2, 3, 4, 5]), requires_grad=True, name='b')
    c = Tensor(np.array([3, 4, 5, 6]), requires_grad=True, name='c')

    d = Add('d')(a, b)
    e = Subtract('e')(c, a)
    output = Add('f')(d, e)

    output.backward()

    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)
    print(e.grad)

    assert np.array_equal(a.grad, np.array([0.] * 4))
    assert np.array_equal(b.grad, np.array([1.] * 4))
    assert np.array_equal(c.grad, np.array([1.] * 4))
    assert np.array_equal(d.grad, np.array([1.] * 4))
    assert np.array_equal(e.grad, np.array([1.] * 4))

def test_Matmul():
    a = Tensor(np.array([1, -1, 1, 1, -1, 1]).reshape((3, 2)),
               requires_grad=True, name='a')

    b = Tensor(np.array([1, 2, 3, 2, 3, 4]).reshape((2, 3)),
               requires_grad=True, name='b')

    c = MatMul('c')(a, b)
    d = Sum('d')(c)
    d.print()

    d.backward()

    a_grad = (np.array([6., 9.] * 3)).reshape((3, 2))
    b_grad = (np.array([1., 1., 1.] * 2)).reshape((2, 3))

    print(a.grad)
    print(b.grad)

    assert np.array_equal(a.grad, a_grad)
    assert np.array_equal(b.grad, b_grad)


def test_ReLU():
    a = Tensor(np.array([1, -1, 2, -2, 3, -3, 4, -4]).reshape((2, 4)),
               requires_grad=True,
               name='a')

    a.print()

    b = relu.ReLU()(a)
    b.print()

    b.backward()
    print(a.grad)

    a_grad = np.array([1, 0, 1, 0, 1, 0, 1, 0]).reshape((2, 4))
    assert np.array_equal(a.grad, a_grad)


def test_Sigmoid():
    a = Tensor(np.array([0.] * 6).reshape((2, 3)),
               requires_grad=True,
               name='a')

    a.print()

    b = sigmoid.Sigmoid()(a)
    b.print()

    b.backward()
    print(a.grad)

    a_grad = np.array([0.25] * 6).reshape((2, 3))
    assert np.array_equal(a.grad, a_grad)


def test_ElementWiseMul():
    a = Tensor(np.array([1, -1, 1, 1, -1, 1]).reshape((3, 2)),
               requires_grad=True, name='a')

    b = Tensor(np.array([1, 2, 2, 3, 3, 4]).reshape((3, 2)),
               requires_grad=True, name='b')

    c = ElementWiseMul('c')(a, b)
    d = Sum('d')(c)
    d.print()

    d.backward()

    a_grad = np.array([[1., 2.], [2., 3.], [3., 4.]])
    b_grad = np.array([[1., -1.], [1., 1.], [-1., 1.]])

    print(a.grad)
    print(b.grad)

    assert np.array_equal(a.grad, a_grad)
    assert np.array_equal(b.grad, b_grad)


def test_BatchElementWiseMul():
    a = Tensor(np.array([1, -1, 1, 1, -1, 1.] * 4).reshape((4, 3, 2)),
               requires_grad=True, name='a')

    b = Tensor(np.array([1, 2, 2, 3, 3, 4.]).reshape((3, 2)),
               requires_grad=True, name='b')

    c = BatchElementWiseMul('c')(a, b)
    d = Sum(name='d')(c)
    d.print()

    d.backward()

    a_grad = np.array([[1., 2.], [2., 3.], [3., 4.]])
    b_grad = np.array([[1., -1.], [1., 1.], [-1., 1.]])

    print(a.grad)
    print(b.grad)

    #assert np.array_equal(a.grad, a_grad)
    #assert np.array_equal(b.grad, b_grad)

def test_Add():

    a = Tensor(requires_grad=True)

    print(isinstance(a.data, np.ndarray))


    b = Tensor()
    #b = Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
    c = Add()(a, b)
    c.print()
    c.backward()
    print(a.grad)
    print(b.grad)

def test_Sum():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    b = Sum(axis=(1, 2))(a)
    b.print()

    b.backward()
    print(a.grad)

def test_Repeat():

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    repeat_time = 3
    target_shape = (6, 3, 4)
    b = Repeat(repeat_time=repeat_time, target_shape=target_shape)(a)

    b.print()
    b.backward()
    print(a.grad)

def test_Get_Set_Sub_Tensor():
    """
    a = np.random.randn(2, 3, 4)
    print(a)

    b = np.arange(6).reshape((1, 2, 3))

    set_sub_ndarray(a, b, ((0, 1), (1, 3), (1, 4)))
    print('\n' * 2)
    print('=' * 20)
    print(a)
    """

    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    coordinate_tuple = ((0, 1), (1, 3), (1, 4))  # shape(1, 2, 3)
    b = Tensor(np.arange(6).reshape(1, 2, 3), requires_grad=True)
    #a.print()

    c = SetSubTensor(coordinate_tuple = coordinate_tuple)(a, b)
    #c.print()

    d = Sum(axis=(1, 2))(c)
    d.backward()
    print(a.grad)

    """
    d = GetSubTensor(coordinate_tuple = coordinate_tuple)(a)
    d.print()
    d.backward()
    print(a.grad)
    """




if __name__ == '__main__':
    # test_Add_Subtract()
    # test_Matmul()
    # test_ReLU()
    # test_Sigmoid()
    # test_ElementWiseMul()

    # test_Add()

    #test_Sum()
    #test_Get_Set_Sub_Tensor()
    #test_BatchElementWiseMul()

    test_Repeat()