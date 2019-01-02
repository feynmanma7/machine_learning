from returing.nn.tensor import Tensor
from returing.nn.operation.base import conv, pad, slide

import numpy as np
np.random.seed(20170430)

def test_sliding2d():
    """
    Input: A (padded) valid Tensor for slides.  [width, height]
    width_idx
    height_idx
    stride

    Output: A sliding Tensor  [K, K], K = kernel_size
    """
    width = 5
    height = 6

    a = Tensor(np.random.randn(width, height), requires_grad=True)
    a.print()

    kernel_size = 2 # Symmetric Squared
    stride = 1

    width_idx = 0
    height_idx = 1

    b = slide.Sliding2D(width_idx=width_idx,
                        height_idx=height_idx,
                        kernel_size=kernel_size,
                        stride=stride)(a)
    #b.print()
    b.backward()
    print(a.grad)


def test_padding2d():

    width = 2
    height = 3

    a = Tensor(np.random.randn(width, height), requires_grad=True)
    a.print()

    padding = 1
    b = pad.Padding2D(padding=padding)(a)
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
    #test_padding2d()
    test_sliding2d()