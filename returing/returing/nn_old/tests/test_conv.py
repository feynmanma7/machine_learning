from returing.nn_old.tensor import Tensor
from returing.nn_old.operation.base import conv, pad, slide

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

def test_convcore2d():

    # input
    n_input_channel = 3
    input_width = 5
    input_height = 5

    # filters
    n_output_channel = 2  # n_filter
    kernel_size = 5
    stride = 1
    padding = 1

    n_samples = 7

    W = Tensor(np.random.randn(kernel_size, kernel_size),
               requires_grad=True)

    X = Tensor(np.random.randn(n_samples,input_width, input_height),
               requires_grad=True)

    Y_pred = conv.ConvCore2D(n_input_channel=n_input_channel,
                         input_width=input_width,
                         input_height=input_height,
                         n_output_channel=n_output_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                             W=W)(X)
    #Y_pred.print()
    Y_pred.backward()
    print(X.grad)

def test_conv2d():

    # input
    n_input_channel = 3
    input_width = 28
    input_height = 28

    # filters
    n_output_channel = 2 # n_filter
    kernel_size = 5
    stride = 2
    padding = 1

    n_samples = 7
    X = Tensor(np.random.randn(n_samples,
                        n_input_channel,
                        input_width, input_height), requires_grad=True)



    Y_pred = conv.Conv2D(n_input_channel=n_input_channel,
                         input_width=input_width,
                         input_height=input_height,
                         n_output_channel=n_output_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)(X)
    #Y_pred.print()
    Y_pred.backward()
    print(X.grad)

    # Y_pred = ReLU()(Y_pred)

    """
    output_width = int((input_width - kernel_size + 2 * padding) / stride + 1)
    output_height = int((input_height - kernel_size + 2 * padding) / stride + 1)

    Y_pred = conv.Conv2D(n_input_channel=n_output_channel,
                         input_width=output_width,
                         input_height=output_height,
                         n_output_channel=n_output_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)(Y_pred)
    #Y_pred.print()
    Y_pred.backward()
    print(X.grad)
    """


if __name__ == '__main__':
    test_conv2d()
    #test_padding2d()
    #test_sliding2d()
    #test_convcore2d()
