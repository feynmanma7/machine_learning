import numpy as np
np.random.seed(20170430)


class Tensor(object):

    data = None
    name = None
    #is_leaf = None
    requires_grad = None
    grad_fn = None # Important, gradient_function is an Operation!
    grad = None
    output_shape = None

    # Point to Tensor
    left_child = None
    right_child = None
    parent = None


    def __init__(self,
                 data=None,
                 requires_grad=False,
                 name=None):
        super(Tensor, self).__init__()
        self.data = data
        self.requires_grad = requires_grad
        self.name = name

    def set_data(self, data):
        self.data = data

    def set_grad(self, grad):
        self.grad = grad

    def backward(self):

        if not self.grad_fn:
            return

        grad_out = None
        if isinstance(self.grad, np.ndarray):
            grad_out = self.grad

        self.grad_fn.backward(grad_out)

        if self.left_child:
            self.left_child.backward()

        if self.right_child:
            self.right_child.backward()

    def print(self, print_data=True):

        if print_data:
            print(self.data)
        if self.requires_grad:
            print('requires_grad =', self.requires_grad)
        if self.grad_fn:
            print('grad_fn = ', self.grad_fn)

        print('\n')


    def _print_graph(self, tensor, parent_to_child=True):

        if parent_to_child:
            if tensor == None:
                return
            print("node.data %s" % tensor.data)
            print("grad_fn %s" % tensor.grad_fn)
            print()

            self._print_graph(tensor.left_child, parent_to_child)
            self._print_graph(tensor.right_child, parent_to_child)

    def print_graph(self, parent_to_child=True):
        if parent_to_child:
            self._print_graph(self, parent_to_child)

