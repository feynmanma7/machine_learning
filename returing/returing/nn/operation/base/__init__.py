from returing.nn.tensor import Tensor
from returing.nn.operation import Operation
import numpy as np
np.random.seed(20170430)


class Add(Operation):

    A = None
    B = None

    def __init__(self):
        super(Add, self).__init__()
        self.A = Tensor()
        self.B = Tensor()

    def forward(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)

        self.A = args[0]
        self.B = args[1]

        C = Tensor(self.A.data + self.B.data)
        C.grad_fn = self
        return C

    def backward(self):
        if self.A.is_grad:
            self.A.grad = 1

        if self.B.is_grad:
            self.B.grad = 1


if __name__ == '__main__':

    a = Tensor(np.array([1, 2, 3, 4]), is_grad=True)
    b = Tensor(np.array([2, 3, 4, 5]), is_grad=False)

    a.print()
    b.print()

    c = Add()(a, b)
    d = Add()(c, a)

    d.backward()
    print(a.grad)
    print(b.grad)

    d.print()
