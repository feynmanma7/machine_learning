from returing.nn.module.module import Module


class Model(object):
    # list of module (or function)
    modules = None

    # For model, use `dict` to store all of the parameters,
    # key: module_idx(or name), value: module.parameters (is a `list`).
    param_dict = None

    def __init__(self,
                 modules=None,
                 n_epoch=None,
                 batch_size=None,
                 verbose=0,
                 loss_fn=None,
                 optimizer=None):
        super(Model, self).__init__()

        self.modules = modules
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.param_dict = {}


    def get_parameters(self):
        # return param_list
        param_list = []
        for name, sub_param_list in self.param_dict.items():
            param_list += sub_param_list

        return param_list

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

    def compile(self,
                n_epoch=None,
                batch_size=None,
                verbose=0,
                loss_fn=None,
                optimizer=None):

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Collect parameters of each sub-module
        for idx, module in enumerate(self.modules):
            if isinstance(module.parameters, list):
                self.param_dict[idx] = module.parameters

    def fit(self, X, y):

        # X: tuple of (batch) attribute data
        # y: (batch) label data

        # batch_inputs = _get_batch(inputs)

        for epoch in range(self.n_epoch):
            y_pred, = self.forward(X)

            loss, = self.loss_fn(y_pred, y)
            if self.verbose == 0:
                print('Epoch:%d, loss:%.4f' % (epoch, loss.data))

            loss.backward()

            param_list = self.get_parameters()
            self.optimizer.step(param_list)



