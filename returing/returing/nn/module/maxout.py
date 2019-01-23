from returing.nn.module.module import Module
from returing.nn.function.base.maximum import Maximum
from returing.nn.function.base.dot_fn import Dot
from returing.nn.function.base.add_fn import BatchAdd
from returing.nn.tensor.parameter import Parameter
from returing.nn.util.initialization import random_init


class Maxout(Module):

    # list of parameter
    parameters = None

    # n_kernel * [input_dim, hidden_dim]
    W_list = None

    # n_kernel * [hidden_dim, ]
    bias_list = None

    def __init__(self,
                 n_kernel=None,
                 input_dim=None,
                 hidden_dim=None):
        super(Maxout, self).__init__()

        # K, number of kernel
        self.n_kernel = n_kernel

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize Parameters
        self.parameters = []
        self.W_list = []
        self.bias_list = []

        for idx in range(self.n_kernel):
            W = Parameter(data=random_init((input_dim, hidden_dim)), name='W'+str(idx))
            self.W_list.append(W)
            self.parameters.append(W)

            bias = Parameter(data=random_init((hidden_dim, )), name='bias'+str(idx))
            self.bias_list.append(bias)
            self.parameters.append(bias)

    def forward(self, inputs):
       """
       # Input
         X: [n_samples, input_dim]

       # Parameters
         W_list: n_kernel * [input_dim, hidden_dim]
         bias_list: n_kernel * [hidden_dim, ]

       # Output
         Y_pred: [n_samples, hidden_dim]

         Y_pred_list: list of Y_pred, n_kernel * [n_samples, hidden_dim]
         Y_pred_i = X \cdot W_i + bias_i, [n_samples, hidden_dim]

         Y_pred = Y_pred_0

         for i in range(1, len(Y_pred_list)):
            Y_pred = Maximum(Y_pred, Y_pred_list[i]) # element-wise maximum

       """
       X, = inputs

       Y_pred, = MaxoutCell()(X, self.W_list, self.bias_list)

       return Y_pred,


class MaxoutCell(Module):

    def __init__(self):
        super(MaxoutCell, self).__init__()

    def forward(self, inputs):
        """
        # Input
        X: [n_samples, input_dim]

        # Parameters
        W_list: n_kernel * [input_dim, hidden_dim]
        bias_list: n_kernel * [hidden_dim, ]

        # Output
        Y_pred: [n_samples, hidden_dim]

        Y_pred_list: list of Y_pred, n_kernel * [n_samples, hidden_dim]
        Y_pred_i = X \cdot W_i + bias_i, [n_samples, hidden_dim]

        Y_pred = Y_pred_0

        for i in range(1, len(Y_pred_list)):
        Y_pred = Maximum(Y_pred, Y_pred_list[i]) # element-wise maximum
        """

        X, W_list, bias_list = inputs

        Y_pred = None
        for idx in range(len(W_list)):
            W_i = W_list[idx]
            bias_i = bias_list[idx]

            # X: [n_samples, input_dim]
            # W_i: [input_dim, hidden_dim]
            # Y_pred_i: [n_samples, hidden_dim]
            Y_pred_i, = Dot()(X, W_i)

            # bias_i: [hidden_dim, ]
            Y_pred_i, = BatchAdd()(Y_pred_i, bias_i)

            if idx == 0:
                Y_pred = Y_pred_i
            else:
                Y_pred, = Maximum()(Y_pred, Y_pred_i)

        return Y_pred,

