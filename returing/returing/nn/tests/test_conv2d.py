from returing.nn.tensor.tensor import Tensor

from returing.nn.module.model import Model

from returing.nn.module.conv.conv2d import Conv2D

from returing.nn.function.conv.padding2d import Padding2D

from returing.nn.function.loss.mse import MSELoss

from returing.nn.optimizer.sgd import SGD

import numpy as np
np.random.seed(20170430)


def test_padding2d():

    n_samples = 2
    n_in_ch = 2
    in_w = 2
    in_h = 2
    padding = 1

    a = Tensor(np.random.randn(n_samples, n_in_ch, in_w, in_h), requires_grad=True, is_leaf=True)
    b, = Padding2D(padding=padding)(a)
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
                    initializer=None,
                    activation=None)

    y_pred, = conv2d(X)

    #print(y_pred.data)
    #y_pred.backward()

    output_width = int((input_width - kernel_size + 2 * padding) / stride + 1)
    output_height = int((input_height - kernel_size + 2 * padding) / stride + 1)
    y = Tensor(np.random.randn(n_samples,
                               n_output_channel,
                               output_width,
                               output_height
                               ))
    loss, = MSELoss()(y_pred, y)
    loss.backward()
    print(conv2d.W.grad.data)
    print(conv2d.bias.grad.data)
    print(loss.data)

def test_model():
    n_samples = 20
    n_input_channel = 3
    input_width = 28
    input_height = 28
    n_output_channel = 4
    stride = 1
    padding = 0
    kernel_size = 5

    X = Tensor(np.random.randn(n_samples,
                               n_input_channel,
                               input_width,
                               input_height))

    output_width = int((input_width - kernel_size + 2 * padding) / stride + 1)
    output_height = int((input_height - kernel_size + 2 * padding) / stride + 1)
    y = Tensor(np.random.randn(n_samples,
                               n_output_channel,
                               output_width,
                               output_height))

    conv2d = Conv2D(n_input_channel=n_input_channel,
                    n_output_channel=n_output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    input_width=input_width,
                    input_height=input_height,
                    is_bias=True,
                    initializer=None,
                    activation=None)

    model = Model()
    model.add(conv2d)

    n_epoch = 10
    batch_size = 2
    verbose = 0
    loss_fn = MSELoss()
    optimizer = SGD(lr=1e-5)

    model.compile(n_epoch=n_epoch,
                batch_size=batch_size,
                verbose=verbose,
                loss_fn=loss_fn,
                optimizer=optimizer)
    model.summary()
    model.fit(X, y)


if __name__ == '__main__':
    #test_conv2d()
    #test_padding2d()
    test_model()