from returing.nn.operation import Operation
from returing.nn.tensor import Tensor

from returing.nn.operation.base import Sum, ElementWiseMul, Add
from returing.nn.operation.base.pad import Padding2D
from returing.nn.operation.base.slide import Sliding2D

from returing.nn.utils import safe_read_dict

import numpy as np
np.random.seed(20170430)


class CrossCorrelation2D(Operation):
    def __init__(self,
                 W = None,
                 input_width=1,
                 input_weight=1,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 name=None):
        super(CrossCorrelation2D, self).__init__()
        self.op_name = 'cross_correlation_2d'
        self.name = name

        self.input_width = input_width
        self.input_height = input_weight
        self.kernel_size = kernel_size # kernel_size
        self.stride = stride # stride
        self.padding = padding # padding

        self.W = W # weights, [K, K]

    def set_weights(self, W):
        # Weights, Kernels, Filters
        assert isinstance(W, Tensor)
        assert W.data.shape == (self.K, self.K)

        self.W = W

    def forward(self, *args):
        """
        # CrossCorrelation2D (Convolution on a 2-D array X, By a kernel W)
        ## Input: X [width, height], W [K, K]
        ## Output: Y [K, K]

        padding_X = Padding2D(padding)(X)  [width + 2P, height + 2P]

        Y[i, j] = Sliding2D(i, j, stride)(padding_X) [K, K]
        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        X = args[0]
        assert X.shape == (self.input_width, self.input_height)

        Y_pred_data = np.zeros(self.W.shape)

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                padding_X = Padding2D(self.padding)(X)
                sub_X = Sliding2D(
                    width_idx=i,
                    height_idx=j,
                    stride=self.stride,
                    kernel_size=self.kernel_size)(padding_X)

                Y_pred_data[i, j] = Sum()(ElementWiseMul(sub_X, self.W))

        Y_pred = Tensor(Y_pred_data)
        return Y_pred


class Conv2D(Operation):
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__()
        self.op_name = 'conv2d'
        self.name = None

        self.n_input_channel = safe_read_dict(kwargs, 'n_input_channel', -1)
        self.input_width = safe_read_dict(kwargs, 'input_width', -1)
        self.input_height = safe_read_dict(kwargs, 'input_height', -1)

        self.n_output_channel = safe_read_dict(kwargs, 'n_output_channel', -1)
        self.output_width = safe_read_dict(kwargs, 'output_width', -1)
        self.output_height = safe_read_dict(kwargs, 'output_height', -1)

        self.kernel_size = safe_read_dict(kwargs, 'kernel_size', -1)
        self.stride = safe_read_dict(kwargs, 'stride', -1)
        self.padding = safe_read_dict(kwargs, 'padding', -1)

        self.initialization = safe_read_dict(kwargs, 'initialization', None)
        self.is_bias = safe_read_dict(kwargs, 'is_bias', False)

        # Initialization



    def forward(self, *args):
        """
        # Dimension Computation Rule
        > output_dim = (N - K + 2P) / S + 1
        > output_dim: output width or height
        > N: input_dim (input width or height)
        > K: filter_size, kernel_size
        > S: stride
        > P: padding

        # Input
        X: [n_samples, n_input_channel, input_width, input_height]

        # Output
        Y: [n_samples, n_output_channel, output_width, output_height]

        # Parameters

        W = n_output_channel * (K * K) [n_output_channel, K, K]
        b = n_output_channel * 1       [n_output, 1]

        total_n_parameters =
            n_output_channel(n_filters) *
                (kernel_size * kernel_size + 1 if is_bias else 0)

        output_width = (input_width - K + 2P) / S + 1
        output_height = (input_height - K + 2P) / S + 1

        # Forward Rule
        Y = ListAdd()([Y_i]),
            i = 0, 1, ..., n_output_channel-1,
            ListAdd iterates over output channels.

        Y_i = ListAdd()([ A_j ]) + b_i
            j = 0, 1, ..., n_input_channel-1,
            ListAdd iterate over input channels.

        A_j = CrossCorrelation2D ( X_j, W_i ), See Wikipedia for cross-correlation.

        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        X = args[0]

        Y_pred = Tensor()

        for i in range(self.n_output_channel):

            Y_i = Tensor()
            for j in range(self.n_input_channel):
                A_j = CrossCorrelation2D()(X[j], self.W[i])
                Y_i = Add()(Y_i, A_j)

            if self.is_bias:
                Y_i = Add()(Y_i, self.b[i])

            Y_pred = Add()(Y_pred, Y_i)

        return Y_pred

    """
    def backward(self, *args, **kwargs):
        pass
    """