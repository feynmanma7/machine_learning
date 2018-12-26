class Operation(object):

    params = None

    def __init__(self, *args, **kwargs):
        super(Operation, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self):
        pass