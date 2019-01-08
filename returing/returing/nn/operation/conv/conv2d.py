from returing.nn.operation.operation import Operation
from returing.nn.tensor.tensor import Tensor
from returing.nn.util import initialization
import numpy as np
np.random.seed()


class Conv2D(Operation):

    # W = None
    # b = None

    def __init__(self,
                 n_input_channel=None,
                 input_width=None,
                 input_height=None,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 n_output_channel=None,
                 is_bias=False):
        super(Conv2D, self).__init__()

        self.n_input_channel = n_input_channel
        self.input_width = input_width
        self.input_height = input_height
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_output_channel = n_output_channel
        self.is_bias = is_bias

        # Initialization
        self.W = initialization.random_init_tensor(
            (n_output_channel, self.kernel_size, self.kernel_size),
            requires_grad=True)

        if self.is_bias:
            self.b = initialization.random_init_tensor(
                (n_output_channel, 1),
                requires_grad=True)

    def forward(self, input_tensor_list):
        assert len(input_tensor_list) == 1
        input_tensor = input_tensor_list[0]
        assert isinstance(input_tensor, Tensor)

        (self.n_samples, n_input_channel, input_width, input_height) = input_tensor.shape()
        assert self.n_input_channel == n_input_channel
        #assert self.input_width == input_width
        #assert self.input_height == input_height

        self.output_width = int((input_width - self.kernel_size + 2 * self.padding)
                           / self.stride + 1)
        self.output_height = int((input_height - self.kernel_size + 2 * self.padding)
                           / self.stride + 1)

        self.output_shape = (self.n_samples,
                        self.n_output_channel,
                        self.output_width,
                        self.output_height)

        #y_pred_data = np.zeros(self.output_shape) # conv_operation
        y_pred_data = self._conv2d_output_channels(input_tensor)

        y_pred = Tensor(y_pred_data)
        y_pred.grad_fn = self
        if input_tensor.requires_grad:
            y_pred.requires_grad = True

        self.input_list = [input_tensor, self.W]
        if self.is_bias:
             self.input_list.append(self.b)
        self.input_requires_grad = input_tensor.requires_grad

        return y_pred

    def backward(self, grad_out):
        if isinstance(grad_out, Tensor):
            assert grad_out.shape() == self.output_shape

        input_grad = None
        if self.input_requires_grad:
            input_grad = Tensor()
        W_grad = Tensor()
        b_grad = Tensor()

        return input_grad, W_grad, b_grad

    def _conv2d_output_channels(self, input_tensor):
        """
        For each output_channel, 
            y_pred_data[:, output_channel, ] =  sum_j^{n_input_channel} 
                conv2d(X[:, input_channel, ], W)
            
        y_pred_data += b
        """

        # TODO
        # Padding of input_tensor

        y_pred_data = np.zeros(self.output_shape)  # conv_operation

        for output_channel_idx in range(self.n_output_channel):
            # [n_samples, output_width, output_hight]
            y_pred_one_channel_data \
                = self._conv2d_output_one_channel(input_tensor, output_channel_idx)
            y_pred_data[:, output_channel_idx, :, :] = y_pred_one_channel_data

        # TODO
        """
        if self.is_bias:
            # y_pred_data [n_samples, 
            y_pred_data += self.b.data
        """

        return y_pred_data

    def _conv2d_output_one_channel(self, input_tensor, output_channel_idx):
        y_pred_one_channel_data = np.zeros(
            (self.n_samples, self.output_width, self.output_height))

        for input_channel_idx in range(self.n_input_channel):
            # [n_samples, output_width, output_height]
            conv2d_input_one_channel_data = self._conv2d_input_one_channel(
                input_tensor, input_channel_idx, output_channel_idx)

            y_pred_one_channel_data += conv2d_input_one_channel_data

        return y_pred_one_channel_data

    def _conv2d_input_one_channel(self, input_tensor, input_channel_idx, output_channel_idx):
        conv2d_input_one_channel_data = np.zeros(
            (self.n_samples, self.output_width, self.output_height))

        for width_idx in range(self.output_width):
            for height_idx in range(self.output_height):
                # conv2d_once: [n_samples, ]
                conv2d_input_one_channel_data[:, width_idx, height_idx] = \
                    self._conv2d_once(input_tensor,
                                      input_channel_idx,
                                      output_channel_idx,
                                      width_idx,
                                      height_idx)

        return conv2d_input_one_channel_data

    def _conv2d_once(self,
                     input_tensor,
                     input_channel_idx,
                     output_channel_idx,
                     width_idx,
                     height_idx):
        width_start = width_idx * self.stride
        width_end = width_idx * self.stride + self.kernel_size

        height_start = height_idx * self.stride
        height_end = height_idx * self.stride + self.kernel_size

        # [n_samples, kernel_size, kernel_size]
        sub_x = input_tensor.data[:,
                input_channel_idx,
                width_start:width_end,
                height_start:height_end]

        # [kernel_size, kernel_size]
        sub_w = self.W.data[output_channel_idx]

        # [n_samples, kernel_size, kernel_size]
        # Use broadcast of numpy.
        conv2d_once_data = sub_x * sub_w

        # [n_samples, ]
        conv2d_once_data = np.sum(conv2d_once_data, axis=(1, 2))

        return conv2d_once_data