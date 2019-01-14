from returing.nn.function.function import Function


#class Module(Function):
class Module(object):

    # tuple of input tensor
    inputs = None

    # tuple of output tensor
    outputs = None

    # `list` of parameter,
    # for tuple.add will create new tuple while list.append not.
    parameters = None

    def __init__(self):
        super(Module, self).__init__()

    def __call__(self, *args):
        return self.forward(args)

    def forward(self, inputs):
        raise NotImplementedError

