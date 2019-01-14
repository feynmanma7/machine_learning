from returing.nn.tensor.tensor import Tensor
from returing.nn.module.conv.conv2d_layer import Conv2D

from returing.nn.function.conv.padding2d_func import Padding2DFunc

import numpy as np
np.random.seed(20170430)


def test_padding2d():

    n_samples = 2
    n_in_ch = 2
    in_w = 2
    in_h = 2
    padding = 1

    a = Tensor(np.random.randn(n_samples, n_in_ch, in_w, in_h), requires_grad=True, is_leaf=True)
    b, = Padding2DFunc(padding=padding)(a)
    print(b.data)

    b.backward()
    print(a.grad.data)



def test_conv2d():

    n_samples = 2
    n_input_channel = 3
    input_width = 28
    input_height = 28
    n_output_channel = 4
    stride = 1
    padding = 0
    kernel_size = 5

    X = Tensor(np.random.randn(n_samples, n_input_channel, input_width, input_height))

    conv2d = Conv2D(n_input_channel=n_input_channel,
                    n_output_channel=n_output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    input_width=input_width,
                    input_height=input_height,
                    is_bias=True,
                    initializer=None)

    y_pred, = conv2d(X)

    #print(y_pred.data)
    y_pred.backward()
    print(conv2d.W.grad.data)
    print(conv2d.bias.grad.data)


if __name__ == '__main__':
    test_conv2d()
    #test_padding2d()