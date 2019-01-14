from returing.nn.tensor.tensor import Tensor


class Parameter(Tensor):

    def __init__(self, data=None, name=None):
        self.data = data
        self.name = name
        self.requires_grad = True
        self.is_leaf = True

