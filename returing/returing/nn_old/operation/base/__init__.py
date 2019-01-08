from returing.nn_old.tensor import Tensor
from returing.nn_old.operation import Operation

from returing.nn_old.utils import set_sub_ndarray, get_sub_ndarray, get_shape_by_coord_tuple

import numpy as np
np.random.seed(20170430)


class Add(Operation):
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

        self.output_shape = C.data.shape

        return C

    def backward(self, C_grad=None):
        if isinstance(self.A.data, np.ndarray):
            shape = self.A.data.shape
        elif isinstance(self.B.data, np.ndarray):
            shape = self.B.data.shape
        else:
            return

        cur_grad = np.ones(shape)

        if isinstance(C_grad, np.ndarray):
            assert C_grad.shape == self.output_shape

            assert cur_grad.shape == C_grad.shape
            cur_grad *= cur_grad

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                # if isinstance(C_grad, np.ndarray),
                # '*' is element-wise multiply
                self.A.grad = cur_grad
            else:
                self.A.grad += cur_grad

        if self.B.requires_grad:
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = cur_grad
            else:
                self.B.grad += cur_grad


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

    def __init__(self, axis=None, target_shape=None, name=None):
        super(Sum, self).__init__()
        self.A = Tensor
        self.op_name = 'sum'
        self.name = name

        self.axis = axis
        self.target_shape = target_shape

    def forward(self, *args):

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        self.A = args[0]

        C_data = np.sum(self.A.data, axis=self.axis)
        if isinstance(self.target_shape, tuple):
            C_data = C_data.reshape(self.target_shape)

        self.output_shape = C_data.shape

        C = Tensor(C_data)

        C.name = self.name
        C.grad_fn = self

        if self.A.requires_grad:
            C.requires_grad = True

        self.A.parent = C
        # self.B.parent = C
        C.left_child = self.A
        #C.right_child = self.B

        return C

    def backward(self, grad_out=None):
        # Add attribute `axis` to the forward function,
        # for backward, the gradient keeps the same with the input Tensor  `A`.

        assert isinstance(self.A.data, np.ndarray)

        if isinstance(grad_out, np.ndarray):
            assert grad_out.shape == self.output_shape
            cur_grad = grad_out # grad_out * ones
        else:
            cur_grad = np.ones(self.output_shape) # ones

        # cur_grad, Reshape to A.shape
        repeat_time = 1
        for dim in self.A.data.shape:
            repeat_time *= dim
        for dim in self.output_shape:
            repeat_time /= dim
        cur_grad = np.repeat(cur_grad, repeat_time).reshape(self.A.data.shape)

        if self.A.requires_grad:
            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = cur_grad
            else:
                self.A.grad += cur_grad


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

            A_cur_grad = self.B.data

            if isinstance(C_grad, np.ndarray):
                # C_grad.shape == A_cur_grad.shape
                A_cur_grad *= C_grad

            if not isinstance(self.A.grad, np.ndarray):
                #self.A.grad = self.B.data
                self.A.grad = A_cur_grad
            else:
                #self.A.grad += self.B.data
                self.A.grad += A_cur_grad

        if self.B.requires_grad:

            B_cur_grad = self.A.data

            if isinstance(C_grad, np.ndarray):
                B_cur_grad *= C_grad

            if not isinstance(self.B.grad, np.ndarray):
                #self.B.grad = self.A.data
                self.B.grad = B_cur_grad
            else:
                #self.B.grad += self.A.data
                self.B.grad += B_cur_grad


class BatchElementWiseMul(Operation):
    """
    Assume A is the batch samples.
    # Input
    A: [n_samples, m, n]
    B: [m, n]

    + Output:
    C: [n_samples, m, n]
    C = A * B, Use the `Broadcast` mechanism of numpy.

    C.grad: [n_samples, m, n]

    grad_A (C) = [n_samples, B]
    grad_B (C) = A / n_samples --> [m, n]
    """

    def __init__(self, name=None):
        super(BatchElementWiseMul, self).__init__()
        self.op_name = 'batch_element_wise_mul'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        # Currrently, A is the batch samples
        assert self.A.data.shape[1:] == self.B.data.shape

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

            # B.data [m, n]
            # A_cur_data [n_samples, m, n]
            n_samples = self.A.data.shape[0]
            A_cur_grad = np.repeat(self.B.data, n_samples).reshape(self.A.data.shape)

            if isinstance(C_grad, np.ndarray):
                # C_grad.shape == A_cur_grad.shape
                A_cur_grad *= C_grad

            if not isinstance(self.A.grad, np.ndarray):
                #self.A.grad = self.B.data
                self.A.grad = A_cur_grad
            else:
                #self.A.grad += self.B.data
                self.A.grad += A_cur_grad

        if self.B.requires_grad:

            # A.data: [n_samples, m, n]
            # B_cur_grad: [m, n]
            n_samples = self.A.data.shape[0]
            assert n_samples > 0

            B_cur_grad = self.A.data

            if isinstance(C_grad, np.ndarray):
                B_cur_grad *= C_grad

            # From [n_samples, m, n] to [m, n]
            # !!! Use numpy.mean( ,axis=0)
            B_cur_grad = np.mean(B_cur_grad, axis=0)
            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = B_cur_grad
            else:
                self.B.grad += B_cur_grad


class SetSubTensor(Operation):
    """
    Set And Get SubTensor is a subset of Add Operation.

    A = Set(coordinate_tuple)(A, B)

    shape = get_shape_by_coord_tuple(coordinate_tuple)

    assert shape == B.shape !!!
    """

    def __init__(self, coordinate_tuple):
        super(SetSubTensor, self).__init__()
        self.coordinate_tuple = coordinate_tuple
        self.shape = get_shape_by_coord_tuple(self.coordinate_tuple)

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        assert self.shape == self.B.data.shape

        C_data = self.A.data
        set_sub_ndarray(C_data, self.B.data, self.coordinate_tuple)

        assert C_data.shape == self.A.data.shape

        C = Tensor(C_data)

        C.left_child = self.A
        C.right_child = self.B

        self.output_shape = C_data.shape

        C.grad_fn = self

        self.A.parent = C
        self.B.parent = C

        if self.A.requires_grad or self.B.requires_grad:
            C.requires_grad = True

        return C

    def backward(self, C_grad=None):
        # The gradient of Operation `SetSubTensor` is 1.

        if isinstance(C_grad, np.ndarray):
            assert C_grad.shape == self.output_shape

        if self.A.requires_grad:
            assert isinstance(self.A.data, np.ndarray)
            if not isinstance(C_grad, np.ndarray):
                A_cur_grad = np.ones(self.A.data.shape) # * cur_grad, cur_grad = ones
            else:
                A_cur_grad = C_grad

            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = np.zeros(self.A.data.shape)
                self.A.grad = A_cur_grad
            else:
                assert self.A.grad.shape == self.A.data.shape
                self.A.grad += A_cur_grad

            assert self.A.grad.shape == self.A.data.shape

        if self.B.requires_grad:
            assert isinstance(self.B.data, np.ndarray)

            if not isinstance(C_grad, np.ndarray):
                B_cur_grad = np.ones(self.B.data.shape) # * cur_grad, cur_grad = ones
            else:
                B_cur_grad = get_sub_ndarray(C_grad, self.coordinate_tuple)

            assert B_cur_grad.shape == self.B.data.shape

            if not isinstance(self.B.grad, np.ndarray):
                self.B.grad = np.zeros(self.B.data.shape)
                self.B.grad = B_cur_grad
            else:
                self.B.grad += B_cur_grad

            assert self.B.grad.shape == self.B.data.shape


class GetSubTensor(Operation):

    def __init__(self, coordinate_tuple):
        super(GetSubTensor, self).__init__()
        self.coordinate_tuple = coordinate_tuple
        self.shape = get_shape_by_coord_tuple(self.coordinate_tuple)

    def forward(self, *args):
        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        self.A = args[0]

        C_data = get_sub_ndarray(self.A.data, self.coordinate_tuple)

        C = Tensor()
        C.data = C_data

        C.left_child = self.A
        # C.right_child = self.B

        C.grad_fn = self

        self.A.parent = C
        #self.B.parent = C

        if self.A.requires_grad:
            C.requires_grad = True

        return C

    def backward(self, C_grad=None):
        if not self.A.requires_grad:
            return

        if not isinstance(self.A.data, np.ndarray):
            return

        if not isinstance(C_grad, np.ndarray):
            C_grad = np.ones(self.shape)

        if not isinstance(self.A.grad, np.ndarray):
            self.A.grad = np.zeros(self.A.data.shape)
            set_sub_ndarray(self.A.grad, C_grad, self.coordinate_tuple, is_add=False)
        else:
            assert self.A.grad.shape == self.A.data.shape
            set_sub_ndarray(self.A.grad, C_grad, self.coordinate_tuple, is_add=True)


class Reshape(Operation):

    def __init__(self, target_shape=(1, )):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        assert isinstance(args[0].data, np.ndarray)

        self.A = args[0]

        self.shape = self.A.data.shape

        C_data = self.A.data.reshape(self.target_shape)

        C = Tensor()
        C.data = C_data

        C.left_child = self.A
        # C.right_child = self.B

        C.grad_fn = self

        self.A.parent = C
        #self.B.parent = C

        if self.A.requires_grad:
            C.requires_grad = True

        return C

    def backward(self, C_grad=None):
        if not self.A.requires_grad:
            return

        if not isinstance(self.A.data, np.ndarray):
            return

        if not isinstance(C_grad, np.ndarray):
            grad_out = np.ones(self.shape)
        else:
            grad_out = C_grad.reshape(self.shape)

        """
        C_grad: target_shape
        
        A.grad: shape
        """

        #cur_grad = np.ones(self.A.data.shape)
        #cur_grad *= grad_out
        cur_grad = grad_out

        if not isinstance(self.A.grad, np.ndarray):
            self.A.grad = cur_grad
        else:
            self.A.grad += cur_grad

        assert self.A.grad.shape == cur_grad.shape



class Repeat(Operation):
    """
    A = Repeat(A, target_shape)

    """

    def __init__(self, repeat_time=-1, target_shape=None):
        super(Repeat, self).__init__()
        self.repeat_time = repeat_time
        self.target_shape = target_shape

    def forward(self, *args):
        assert len(args) == 1
        assert isinstance(args[0], Tensor)
        assert self.repeat_time > 0

        self.A = args[0]

        C_data = np.repeat(self.A.data, self.repeat_time)

        if self.target_shape:
            C_data = C_data.reshape(self.target_shape)

        C = Tensor(C_data)
        C.grad_fn = self
        C.left_child = self.A
        self.A.parent = C
        self.output_shape = C_data.shape
        if self.A.requires_grad:
            C.requires_grad = True

        return C

    def backward(self, C_grad=None):
        # The gradient of Operation `Repeat` is 1.

        if isinstance(C_grad, np.ndarray):
            assert C_grad.shape == self.output_shape

        if self.A.requires_grad:
            assert isinstance(self.A.data, np.ndarray)

            if not isinstance(C_grad, np.ndarray):
                A_cur_grad = np.ones(self.A.data.shape) # * cur_grad, cur_grad = ones
            else:
                A_cur_grad = C_grad.reshape(self.A.data.shape)

            assert A_cur_grad.shape == self.A.data.shape

            if not isinstance(self.A.grad, np.ndarray):
                self.A.grad = np.zeros(self.A.data.shape)
                self.A.grad = A_cur_grad
            else:
                assert self.A.grad.shape == self.A.data.shape
                self.A.grad += A_cur_grad

            assert self.A.grad.shape == self.A.data.shape
