from ..tensor.tensor import Tensor


class Optimizer(Tensor):

    params = None

    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError