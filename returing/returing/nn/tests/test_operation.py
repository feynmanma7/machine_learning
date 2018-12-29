import numpy as np
np.random.seed(20170430)

from returing.nn.operation.base import *
from returing.nn.operation.activation import relu, sigmoid


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


if __name__ == '__main__':
    # test_Add_Subtract()
    # test_Matmul()
    # test_ReLU()
    # test_Sigmoid()
    # test_ElementWiseMul()

    test_Add()