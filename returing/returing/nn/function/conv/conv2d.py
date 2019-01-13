from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Conv2DFunc(Function):

    def __init__(self, **kwargs):
        super(Conv2DFunc, self).__init__()

        self.n_input_channel = kwargs['n_input_channel']
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.padding = kwargs['padding']
        self.n_output_channel = kwargs['n_output_channel']
        self.is_bias = kwargs['is_bias']

    def forward(self, inputs):
        X, W, bias = inputs

        (self.n_samples, n_input_channel, input_width, input_height) = X.shape()

        self.output_width = int((input_width - self.kernel_size + 2 * self.padding)
                                / self.stride + 1)
        self.output_height = int((input_height - self.kernel_size + 2 * self.padding)
                                 / self.stride + 1)

        self.output_shape = (self.n_samples,
                             self.n_output_channel,
                             self.output_width,
                             self.output_height)

        bias_requires_grad = bias.requires_grad if isinstance(bias, Tensor) else False

        self.saved_context = X.requires_grad, W.requires_grad, bias_requires_grad

        # X: [n_samples, n_in_ch, in_w, in_h]
        # padding_X: [n_samples, n_in_ch, in_w+2p, in_h+2p]
        padding_X = Paddding2DFunc()(X)

        y_pred = Tensor()

        for oC_idx in range(self.n_output_channel):
            # W: [n_out_ch, n_in_ch, K, K]
            # W_oC: [n_in_ch, K, K]
            coord_tuple = ((oC_idx, oC_idx+1),
                           (0, self.n_input_channel),
                           (0, self.kernel_size),
                           (0, self.kernel_size))
            W_oC = GetSubTensorFunc(coord_tuple=coord_tuple)(W)

            for w_idx in range(self.output_width):
                for h_idx in range(self.output_height):

                    # X:[n_samples, n_in_ch, in_w+2p, in_h+2p]
                    coord_tuple = ((0, self.n_samples),
                                   (0, self.n_input_channel),
                                   (w_idx*self.stride, w_idx*self.stride + self.kernel_size),
                                   (h_idx*self.stride, h_idx*self.stride + self.kernel_size))

                    # sub_X: [n_samples, n_in_ch, K, K]
                    sub_X = GetSubTensorFunc(coord_tuple=coord_tuple)(padding_X)

                    # x: [n_samples, ]
                    x = BatchMatMulFunc()(sub_X, W_oC)

                    # [n_out_ch, out_w, out_h, n_samples]
                    y_pred = ArrayStackFunc()(y_pred, x) # !!!
                    # coord_tuple = ((), (), (), ())
                    # y_pred = SetSubTensorFunc()(y_pred, x)

        target_shape = (self.n_samples,
                        self.n_output_channel,
                        self.output_width,
                        self.output_height)
        y_pred = ReshapeFunc(target_shape=target_shape)(y_pred)

        return y_pred,

        """
        y_pred_data = self._conv2d_output_channels(X, W, bias)

        y_pred = Tensor(y_pred_data)

        return y_pred,
        """

    def backward(self, grads):
        y_pred_grad, = grads

        W_grad_data = np.zeros(
            (self.n_output_channel,
             self.n_input_channel,
             self.kernel_size,
             self.kernel_size))
        bias_grad_data = np.zeros((self.n_output_channel, ))

        #return input_grad, W_grad, b_grad



        X_requires_grad, W_requires_grad, bias_requires_grad = \
            self.saved_context

        if X_requires_grad:
            # X:[n_samples, n_input_channel, input_width, input_height]

            X_grad_data = np.

    def _conv2d_output_channels(self, input_tensor, W, bias):
        """
        # ===Forward
            For each output_channel,
                y_pred[:, output_channel, :, :] =
                    conv2d(X[:, :, :, :], W[output_channel, :, :, :])

                conv2d(X[:, :, :, :], W_o[:, :, :])[:, i, j] =
                    X[:, :, i*S:i*S+K, j*S:j*S+K] * W_o[:, :, :]

            y_pred: [n_samples, n_output_channel, output_width, output_height]
            b: [n_output_channel]

            y_pred += b
                ==> y_pred_data +=
                    b = b.reshape((n_output_channel, 1, 1))

        # ===Backward
        ## Bias
        y_pred = y_pred + b
        grad_b = np.ones((n_output_channel))

        If grad_out: [n_samples, output_channel, output_width, output_height]
            grad_b *= np.sum(grad_out, axis=(0, 1, 2)), [n_samples, ]

        ## Weights
        y_pred[:, output_channel, :, :] =
            conv2d(X[:, :, :, :], W[output_channel, :, :, :])

        y_pred[:, c, i, j] =
            X[:, :, i*S:i*S+K, j*S:j*S+K] * W[c, :, :, :]

        grad_W[c, :, :, :] += X[:, :, i*S:i*S+K, j*S:j*S+K]

        grad_W

        """

        # TODO
        # Padding of input_tensor

        y_pred_data = np.zeros(self.output_shape)  # conv_operation

        for output_channel_idx in range(self.n_output_channel):
            # y_pred_one_channel_data: [n_samples, output_width, output_hight]
            y_pred_one_channel_data \
                = self._conv2d_output_one_channel(input_tensor, output_channel_idx, W, bias)

            y_pred_data[:, output_channel_idx, :, :] = y_pred_one_channel_data

        # TODO
        """
        if self.is_bias:
            # y_pred_data [n_samples, 
            y_pred_data += self.b.data
        """

        return y_pred_data

    def _conv2d_output_one_channel(self, input_tensor, output_channel_idx, W, bias):
        # [n_samples, output_width, output_height]
        y_pred_one_channel_data = self._conv2d_input_channel(
            input_tensor, output_channel_idx, W, bias)

        return y_pred_one_channel_data

    def _conv2d_input_channel(self, input_tensor, output_channel_idx, W, bias):
        conv2d_input_channel_data = np.zeros(
            (self.n_samples, self.output_width, self.output_height))

        for width_idx in range(self.output_width):
            for height_idx in range(self.output_height):
                # conv2d_once: [n_samples, ]
                conv2d_input_channel_data[:, width_idx, height_idx] = \
                    self._conv2d_once(input_tensor,
                                      output_channel_idx,
                                      width_idx,
                                      height_idx, W, bias)

        return conv2d_input_channel_data

    def _conv2d_once(self,
                     input_tensor,
                     output_channel_idx,
                     width_idx,
                     height_idx, W, bias):
        width_start = width_idx * self.stride
        width_end = width_idx * self.stride + self.kernel_size

        height_start = height_idx * self.stride
        height_end = height_idx * self.stride + self.kernel_size

        # [n_samples, n_input_channel, kernel_size, kernel_size]
        sub_x = input_tensor.data[:, :,
                width_start:width_end,
                height_start:height_end]

        # [n_input_channel, kernel_size, kernel_size]
        sub_w = W.data[output_channel_idx]

        # [n_samples, n_input_channel, kernel_size, kernel_size]
        # Use broadcast of numpy.
        conv2d_once_data = sub_x * sub_w

        # [n_samples, ]
        conv2d_once_data = np.sum(conv2d_once_data, axis=(1, 2, 3))

        return conv2d_once_data