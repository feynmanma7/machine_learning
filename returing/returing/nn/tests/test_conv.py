from returing.nn.tensor import Tensor
from returing.nn.operation.base import conv, padding_op

import numpy as np
np.random.seed(20170430)


def test_padding2d():

    width = 2
    height = 3
    padding = 1
    a = Tensor(np.random.randn(width, height), requires_grad=True)
    a.print()
    b = padding_op.Padding2D(padding=padding)(a)
    b.print()

    b.backward()
    print(a.grad)


def test_conv2d():

    # input
    n_input_channel = 3
    n_input_width = 28
    n_input_height = 28


    # filters
    n_output_channel = 5 # n_filter
    kernel_size = 5
    stride = 2
    padding = 1

    # output: n_output = (N - K + 2P) / S + 1
    n_output_width = int((n_input_width - kernel_size + 2 * padding) / stride) + 1
    n_output_height = int((n_input_height - kernel_size + 2 * padding) / stride) + 1

    n_samples = 10
    X = np.random.randn(n_samples,
                        n_input_channel,
                        n_input_width, n_input_height)

    Y_pred = conv.Conv2D()(X)


if __name__ == '__main__':
    #test_conv2d()
    test_padding2d()