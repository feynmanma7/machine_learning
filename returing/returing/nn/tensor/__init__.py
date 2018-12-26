import numpy as np
np.random.seed(20170430)


class Tensor(object):

    data = None
    is_grad = None # has_grad or not
    grad_fn = None
    grad = None

    def __init__(self,
                 data=None,
                 is_grad=False):
        super(Tensor, self).__init__()
        self.data = data
        self.is_grad = is_grad

    def set_data(self, data):
        self.data = data

    def set_grad(self, grad):
        self.grad = grad

    def backward(self):
        return self.grad_fn.backward()

    def print(self):
        print(self.data)

        if self.is_grad:
            print('is_grad =', self.is_grad)
        if self.grad_fn:
            print('grad_fn = ', self.grad_fn)

        print('\n')
