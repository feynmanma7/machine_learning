from returing.nn.module.module import Module
from returing.nn.tensor.tensor import Tensor
from returing.nn.tensor.parameter import Parameter
from returing.nn.util import initialization
from returing.nn.util.initialization import random_init
import returing.nn.function as F
import numpy as np
np.random.seed()


class Conv2D(Module):

    # Parameter, [n_output_channel, n_input_channel, K, K]
    W = None

    # Parameter, [n_output_channel, ]
    bias = None

    def __init__(self, **kwargs):
        super(Conv2D, self).__init__()

        """
        n_input_channel=None,
                 input_width=None,
                 input_height=None,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 n_output_channel=None,
                 initializer = None,
                 is_bias=True
        
        self.n_input_channel = n_input_channel
        self.input_width = input_width
        self.input_height = input_height
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_output_channel = n_output_channel
        #self.initializer = initializer
        self.is_bias = is_bias
        """

        self.n_input_channel = kwargs['n_input_channel']
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.padding = kwargs['padding']
        self.n_output_channel = kwargs['n_output_channel']
        self.is_bias = kwargs['is_bias']

        initializer = kwargs['initializer']

        self.conv2d_func = F.conv.conv2d.Conv2D(**kwargs)

        # Initialization
        if not initializer:
            initializer = random_init

        self.W = Parameter(data=initializer(
            (self.n_output_channel,
             self.n_input_channel,
             self.kernel_size, self.kernel_size)))

        if self.is_bias:
            self.bias = Parameter(data=initializer(
                (self.n_output_channel, )))

    def forward(self, inputs):
        X, = inputs

        return self.conv2d_func(X, self.W, self.bias)





