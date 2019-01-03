from returing.nn.operation import Operation
from returing.nn.tensor import Tensor

from returing.nn.operation.base import Sum, BatchElementWiseMul, Add, \
    GetSubTensor, SetSubTensor, Reshape
from returing.nn.operation.base.pad import Padding2D
from returing.nn.operation.base.slide import Sliding2D

from returing.nn.utils import safe_read_dict, initialization

import numpy as np
np.random.seed(20170430)


class ConvCore2D(Operation):
    # The basic one-time Convolutional Operation.

    def __init__(self, *args, **kwargs):
        super(ConvCore2D, self).__init__()
        self.op_name = 'cross_correlation_2d'
        #self.name = name
        self.kwargs = kwargs['kwargs']

        self.input_width = safe_read_dict(self.kwargs, 'input_width', 1)
        self.input_height = safe_read_dict(self.kwargs, 'input_height', 1)
        self.kernel_size = safe_read_dict(self.kwargs, 'kernel_size', 1)
        self.stride = safe_read_dict(self.kwargs, 'stride', 1)
        self.padding = safe_read_dict(self.kwargs, 'padding', 0)

        self.W = safe_read_dict(self.kwargs, 'W', None) # weights, [K, K]


    def set_weights(self, W):
        # Weights, Kernels, Filters
        assert isinstance(W, Tensor)
        assert W.data.shape == (self.K, self.K)

        self.W = W

    def forward(self, *args, **kwargs):
        """
        # ConvCore2D (Convolution on a 2-D array of X.data, By a kernel W)
        ## Input: X, [n_samples, input_width, input_height],
        ## Output: Y [n_samples, output_width, output_height]

        W [K, K], weights of one channel.

        padding_X = Padding2D(padding, channel_idx)(X)
            [n_samples, input_width + 2P, input_height + 2P]

        Y_ij = Sliding2D(i, j, stride, channel_idx)(padding_X)
            [K, K]

        Y = SetSubTensor(i, j)(Y_ij)
        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        X = args[0]
        #assert X.data.shape[1:] == (self.input_width, self.input_height)
        assert isinstance(self.W, Tensor)
        assert isinstance(self.W.data, np.ndarray)
        assert self.W.data.shape == (self.kernel_size, self.kernel_size)

        output_width = int((self.input_width - self.kernel_size + 2 * self.padding) \
                            / self.stride + 1)
        output_height = int((self.input_height - self.kernel_size + 2 * self.padding) \
                             / self.stride + 1)

        n_samples = X.data.shape[0]

        # Y_pred: [n_samples, output_width, output_height]
        Y_pred = Tensor(np.zeros((n_samples, output_width, output_height)))

        # X: [n_samples, input_width, input_height]
        # padding_X: [n_samples, input_width+2P, input_weight+2P]
        padding_X = Padding2D(padding=self.padding)(X)

        for i in range(output_width):
            for j in range(output_height):

                # sub_X: [n_samples, K, K]
                sub_X = Sliding2D(
                    width_idx=i,
                    height_idx=j,
                    stride=self.stride,
                    kernel_size=self.kernel_size)(padding_X)

                # sub_X: [n_samples, K, K]
                # W: [K, K]
                # Y_pred_ij: [n_samples, K, K]
                # Rely on Right-align Broadcast of numpy in `ElementWiseMul`.
                Y_pred_ij = BatchElementWiseMul()(sub_X, self.W)

                # Y_pred_ij: [n_samples, 1], actually is `(n_samples, )` of np.sum().
                # Must be reshaped before used !!!
                Y_pred_ij = Sum(axis=(1, 2))(Y_pred_ij)

                # Y_pred: [n_samples, output_width, output_height]
                # Y_pred_ij: (n_samples, ) Reshape To [n_samples, 1]
                coord_tuple = ((0, n_samples),
                               (i, i+1),
                               (j, j+1))
                Y_pred = SetSubTensor(coord_tuple)(Y_pred, Y_pred_ij)

        return Y_pred


class Conv2D(Operation):
    def __init__(self, **kwargs):
        super(Conv2D, self).__init__()
        self.op_name = 'conv2d'
        self.name = None
        self.kwargs = kwargs

        self.n_input_channel = safe_read_dict(kwargs, 'n_input_channel', 1)
        self.input_width = safe_read_dict(kwargs, 'input_width', 1)
        self.input_height = safe_read_dict(kwargs, 'input_height', 1)
        self.n_output_channel = safe_read_dict(kwargs, 'n_output_channel', 1)
        self.kernel_size = safe_read_dict(kwargs, 'kernel_size', 1)
        self.stride = safe_read_dict(kwargs, 'stride', 1)
        self.padding = safe_read_dict(kwargs, 'padding', 0)

        self.initialization = safe_read_dict(kwargs, 'initialization', None)
        self.is_bias = safe_read_dict(kwargs, 'is_bias', True)

        # Initialization
        self.W = initialization.random_init_tensor(
            (self.n_output_channel,
             self.kernel_size,
             self.kernel_size))

        if self.is_bias:
            self.b = initialization.random_init_tensor(
                (self.n_output_channel, 1))

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


        # ===============================Important!!!================================
        # ===== Generated Process
        Y = ListAdd()([Y_i]),
            i = 0, 1, ..., n_output_channel-1,
            ListAdd iterates over output channels,
                (ListAdd is implemented by using Add iteratively)

        Y_i = ListAdd()([ A_j ]) + b_i
            j = 0, 1, ..., n_input_channel-1,
            ListAdd iterate over input channels.

        A_j = ConvCore2D ( X_j, W_i ), See Wikipedia for cross-correlation.

        # ===== Forward Rule

        """

        assert len(args) == 1
        assert isinstance(args[0], Tensor)

        X = args[0]  #[n_samples, n_input_channel, input_width, input_height]

        self.n_samples = X.data.shape[0]

        self.output_width = int((self.input_width - self.kernel_size + 2 * self.padding) \
                            / self.stride + 1)
        self.output_height = int((self.input_height - self.kernel_size + 2 * self.padding) \
                             / self.stride + 1)

        # [n_samples, n_output_channel, output_width, output_height]
        Y_pred = Tensor(np.zeros((self.n_samples,
                                  self.n_output_channel,
                                  self.output_width,
                                  self.output_height)), requires_grad=True)

        for i in range(self.n_output_channel):

            Y_i = Tensor(np.zeros((self.n_samples,
                                   self.output_width,
                                   self.output_height)), requires_grad=True)

            for j in range(self.n_input_channel):
                # X: [n_samples, n_input_channel, input_width, input_height]
                # X_j: [n_samples, 1, input_width, input_height]
                coord_tuple = ((0, self.n_samples),
                               (j, j+1),
                               (0, self.input_width),
                               (0, self.input_height))
                X_j = GetSubTensor(coord_tuple)(X)

                # X_j: [n_samples, 1, input_width, input_height]
                # X_j(Reshaped): [n_samples, input_width, input_height]
                target_shape = (self.n_samples, self.input_width, self.input_height)
                X_j = Reshape(target_shape=target_shape)(X_j)

                # W: [n_output_channel, K, K]
                # W_i: [1, K, K]
                coord_tuple = ((i, i+1),
                               (0, self.kernel_size),
                               (0, self.kernel_size))
                W_i = GetSubTensor(coord_tuple)(self.W)
                target_shape = (self.kernel_size, self.kernel_size)
                W_i = Reshape(target_shape=target_shape)(W_i)

                # A_j: [n_samples, output_width, output_height]
                # W_i: [K, K]
                self.kwargs['W'] = W_i
                A_j = ConvCore2D(kwargs=self.kwargs)(X_j)
                #A_j = ConvCore2D()(X_j)

                # Y_i: [n_samples, output_width, output_height]
                # A_j: [n_samples, output_width, output_height]
                Y_i = Add()(Y_i, A_j)

            if self.is_bias:
                # b: [n_output_channel, 1]
                # b_i: [1, 1]
                coord_tuple = ((i, i+1),
                               (0, 1))
                b_i = GetSubTensor(coord_tuple)(self.b)

                # Y_i: [n_samples, output_width, output_height]
                # b_i: [1, 1]
                # Rely on broadcast of numpy in `Add`.
                # Here b_i is [1, 1], `Reshape` is not needed.
                Y_i = Add()(Y_i, b_i)

            # Y_i: [n_samples, output_width, output_height]
            # Y_pred = [n_sample, n_output_channel, output_width, output_height]
            coord_tuple = ((0, self.n_samples),
                           (i, i+1),
                           (0, self.output_width),
                           (0, self.output_height))
            Y_pred = SetSubTensor(coord_tuple)(Y_pred, Y_i)

        return Y_pred
