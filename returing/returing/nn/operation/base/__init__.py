from returing.nn.tensor import Tensor
from returing.nn.operation import Operation
import numpy as np
np.random.seed(20170430)


class Add(Operation):

    A = None
    B = None

    def __init__(self, name):
        super(Add, self).__init__()
        self.A = Tensor
        self.B = Tensor
        self.op_name = 'add'
        self.name = name

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

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

    def backward(self, grad_out=1):
        if self.A.requires_grad:
            if not self.A.grad:
                self.A.grad = 1 * grad_out
            else:
                self.A.grad += 1 * grad_out

        if self.B.requires_grad:
            if not self.B.grad:
                self.B.grad = 1 * grad_out
            else:
                self.B.grad += 1 * grad_out


if __name__ == '__main__':

    a = Tensor(np.array([1, 2, 3, 4]), requires_grad=True, name='a')
    b = Tensor(np.array([2, 3, 4, 5]), requires_grad=True, name='b')

    a.print()
    b.print()

    c = Add('c')(a, b)
    d = Add('d')(a, b)
    e = Add('e')(d, c)

    e.print()

    e.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)





