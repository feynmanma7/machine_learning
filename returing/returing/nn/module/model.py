#from returing.nn.module.module import Module
from returing.nn.tensor.tensor import Tensor
from queue import Queue

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

    def summary(self):
        for idx, _module in enumerate(self.modules):
            print(idx, '-->', _module)

    def get_parameters(self):
        # Return `param_list`.
        # param_dict is set after compile,
        #   when all of the sub-modules have been set or added.

        # module queue
        q = Queue()

        param_list = []

        for _module in self.modules:
            q.put(_module)

        while not q.empty():
            _module = q.get()

            if isinstance(_module.parameters, list):
                param_list += _module.parameters

            if _module.child_modules:
                for _child_module in _module:
                    q.put(_child_module)

        """
        for name, sub_param_list in self.param_dict.items():
            param_list += sub_param_list
        """

        return param_list

    def add(self, _module):
        # For a model object, sub-module is added sequentially,
        #   sub-module of sub-module is ignored,
        #   while must be registered for parameters.

        if isinstance(self.modules, list):
            self.modules.append(_module)
        else:
            self.modules = [_module]

        """
        if not isinstance(self.modules, list):
            self.modules = []

        self.modules += _module.modules
        """

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

        """
        # Collect parameters of each sub-module recursively.
        for idx, module in enumerate(self.modules):

            if isinstance(module.parameters, list):
                self.param_dict[idx] = module.parameters
        """

    def batch_input_generator(self, X, y):
        # In the future, from file or other source.
        n_samples = X.data.shape[0]
        batch_size = self.batch_size
        total_batch = int(n_samples / batch_size)

        for batch_idx in range(total_batch):
            yield Tensor(X.data[batch_idx*batch_size:(batch_idx+1)*batch_size]), \
                  Tensor(y.data[batch_idx*batch_size:(batch_idx+1)*batch_size])

    def fit_batch(self, X_batch, y_batch):
        pass


    def fit(self, X, y):

        # X: tuple of (batch) attribute data
        # y: (batch) label data

        # batch_inputs = _get_batch(inputs)

        for epoch in range(self.n_epoch):
            idx = 0
            for X_batch, y_batch in self.batch_input_generator(X, y):
                idx += 1
                #y_pred_batch, = self.fit_batch(X_batch, y_batch)
                y_pred_batch, = self.forward(X_batch)

                #y_pred, = self.forward(X)

                #loss, = self.loss_fn(y_pred, y)
                loss, = self.loss_fn(y_pred_batch, y_batch)
                if self.verbose == 0:
                    print('Epoch:%d, batch:%d, loss:%.4f' % (epoch, idx, loss.data))

                loss.backward()

                param_list = self.get_parameters()
                self.optimizer.step(param_list)



