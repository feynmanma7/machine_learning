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

    n_samples = 2
    width = 4
    height = 5

    a = Tensor(np.random.randn(n_samples, width, height), requires_grad=True)
    a.print()

    kernel_size = 2 # Symmetric Squared
    stride = 1

    width_idx = 0
    height_idx = 1

    b = slide.Sliding2D(width_idx=width_idx,
                        height_idx=height_idx,
                        kernel_size=kernel_size,
                        stride=stride)(a)
    b.print()
    b.backward()
    print(a.grad)


def test_padding2d():

    n_samples = 4
    width = 2
    height = 3

    a = Tensor(np.random.randn(n_samples, width, height), requires_grad=True)
    a.print()

    padding = 1
    b = pad.Padding2D(padding=padding)(a)
    b.print()

    b.backward()
    print(a.grad)


def test_conv2d():

    # input
    n_input_channel = 1
    input_width = 5
    input_height = 5


    # filters
    n_output_channel = 1 # n_filter
    kernel_size = 5
    stride = 1
    padding = 0

    n_samples = 1
    X = Tensor(np.random.randn(n_samples,
                        n_input_channel,
                        input_width, input_height), requires_grad=True)

    Y_pred = conv.Conv2D(n_input_channel=n_input_channel,
                         input_width=input_width,
                         input_height=input_height,
                         output_channel=n_output_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)(X)
    Y_pred.print()

    Y_pred.backward()
    print(X.grad)


if __name__ == '__main__':
    test_conv2d()
    #test_padding2d()
    #test_sliding2d()
