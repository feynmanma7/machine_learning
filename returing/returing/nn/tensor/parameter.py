from returing.nn.tensor.tensor import Tensor


class Parameter(Tensor):

    def __init__(self, data=None):
        self.data = data
        self.requires_grad = True
        self.is_leaf = True

