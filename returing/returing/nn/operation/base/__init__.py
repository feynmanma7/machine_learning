from returing.nn.tensor import Tensor
from returing.nn.operation import Operation
import numpy as np
np.random.seed(20170430)


class Add(Operation):

    A = None
    B = None

    def __init__(self, name=None):
        super(Add, self).__init__()
        #self.A = Tensor
        #self.B = Tensor
        self.op_name = 'add'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        # May not have the same shape, use broadcast instead.
        # assert self.A.data.shape == self.B.data.shape

        if not isinstance(self.A.data, np.ndarray):
            C = Tensor(self.B.data)
        elif not isinstance(self.B.data, np.ndarray):
            C = Tensor(self.A.data)
        else:
            C = Tensor(self.A.data + self.B.data)

        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad or self.B.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        self.B.parent = C
        C.left_child = self.A
        C.right_child = self.B

        return C

    def backward(self, C_grad=1):

        # assert self.A.data.shape == self.B.data.shape

        if isinstance(self.A.data, np.ndarray):
            shape = self.A.data.shape
        elif isinstance(self.B.data, np.ndarray):
            shape = self.B.data.shape
        else:
            return

        #grad_mat = np.ones(self.A.data.shape)
        grad_mat = np.ones(shape)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                # if isinstance(C_grad, np.ndarray),
                # '*' is element-wise multiply
                self.A.grad = grad_mat * C_grad
            else:
                self.A.grad += grad_mat * C_grad

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = grad_mat * C_grad
            else:
                self.B.grad += grad_mat * C_grad


class Subtract(Operation):

    A = None
    B = None

    def __init__(self, name=None):
        super(Subtract, self).__init__()
        self.A = Tensor
        self.B = Tensor
        self.op_name = 'subtract'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        assert self.A.data.shape == self.B.data.shape

        C = Tensor(self.A.data - self.B.data)
        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad or self.B.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        self.B.parent = C
        C.left_child = self.A
        C.right_child = self.B

        return C

    def backward(self, C_grad=1):
        assert self.A.data.shape == self.B.data.shape

        grad_mat = np.ones(self.A.data.shape)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = grad_mat * C_grad
            else:
                self.A.grad += grad_mat * C_grad

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = - grad_mat * C_grad
            else:
                self.B.grad += (- grad_mat * C_grad)


class Sum(Operation):
    # Unary operation
    A = None

    def __init__(self, name=None):
        super(Sum, self).__init__()
        self.A = Tensor
        self.op_name = 'sum'
        self.name = name

    def forward(self, *args):

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        self.A = args[0]

        C = Tensor(np.sum(self.A.data))

        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        # self.B.parent = C
        C.left_child = self.A
        #C.right_child = self.B

        return C

    def backward(self, C_grad=1):

        if self.A == None:
            return

        assert isinstance(self.A.data, np.ndarray)

        grad_mat = np.ones(self.A.data.shape)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = grad_mat * C_grad
            else:
                self.A.grad += grad_mat * C_grad


class MatMul(Operation):
    """
    We use the same way as numpy.
    # === dot, matrix multiply
    # === multiply, element-wise multiply

    A: [m, k]
    B: [k, n]
    C = matmul(A, B) : [m, n]

    Jacobbian Matrix

    C.grad: [m, n]

    grad_A (C) = np.dot(C.grad, B.T) np.dot([m, n], [n, k]) ==> [m, k]
    grad_B (C) = np.dot(A.T, C.grad) np.dot([k, m], [m, n]) ==> [k, n]

    """

    A = None # [m, k]
    B = None # [k, n]

    def __init__(self, name=None):
        super(MatMul, self).__init__()
        #self.A = Tensor
        #self.B = Tensor
        self.op_name = 'matmul'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        assert self.A.data.shape[1] == self.B.data.shape[0]

        C = Tensor(np.dot(self.A.data, self.B.data))
        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad or self.B.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        self.B.parent = C
        C.left_child = self.A
        C.right_child = self.B

        return C

    def backward(self, C_grad=1):
        assert self.A.data.shape[1] == self.B.data.shape[0]

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = np.dot(C_grad, self.B.data.T)
            else:
                self.A.grad += np.dot(C_grad, self.B.data.T)

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = np.dot(self.A.data.T, C_grad)
            else:
                self.B.grad += np.dot(self.A.data.T, C_grad)


class ElementWiseMul(Operation):
    """
    We use the same way as numpy.
    # === dot, matrix multiply
    # === multiply, element-wise multiply

    A: [m, n]
    B: [m, n]
    C = ElementWiseMul(A, B) : [m, n]

    C.grad: [m, n]

    grad_A (C) = B
    grad_B (C) = A

    """

    A = None # [m, n]
    B = None # [m, n]

    def __init__(self, name=None):
        super(ElementWiseMul, self).__init__()
        self.op_name = 'element_wise_mul'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        assert self.A.data.shape == self.B.data.shape

        C = Tensor(self.A.data * self.B.data) # In numpy, * means element-wise multiply
        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad or self.B.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        self.B.parent = C
        C.left_child = self.A
        C.right_child = self.B

        return C

    def backward(self, C_grad=None):
        # assert self.A.data.shape == self.B.data.shape

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = self.B.data
            else:
                self.A.grad += self.B.data

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = self.A.data
            else:
                self.B.grad += self.A.data






