from returing.nn.module.module import Module


class Sequential(Module):
    # list of module (or function)
    modules = None

    def __init__(self, modules=None):
        super(Sequential, self).__init__()
        self.modules = modules

    def add(self, _module):
        if isinstance(self.modules, list):
            self.modules.append(_module)
        else:
            self.modules = [_module]

    def forward(self, inputs):
        y_pred = inputs
        for _module in self.modules:
            y_pred = _module(y_pred)

        return y_pred

