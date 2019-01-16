from returing.nn.tensor.tensor import Tensor
from returing.nn.module.rnn import rnn_layer
from returing.nn.function.loss.mse import MSELoss
from returing.nn.function.base.concat import Concat


import numpy as np


def test_rnn():

    n_samples = 2
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    n_time_step = 5
    batch_size = 10
    is_output_bias = True
    is_hidden_bias = True
    verbose = 0
    is_return_sequences = True

    inputs = []
    targets = []
    for _ in range(n_time_step):
        input_tensor = Tensor(np.random.rand(n_samples, n_time_step, input_dim),
                              requires_grad=True,
                              is_leaf=True)
        inputs.append(input_tensor)

        target_tensor = Tensor(np.random.rand(n_samples, n_time_step, output_dim))
        targets.append(target_tensor)

    inputs = tuple(inputs)
    targets = tuple(targets)

    outputs = rnn_layer.RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_time_step=n_time_step,
        batch_size=batch_size,
        is_hidden_bias=is_hidden_bias,
        is_output_bias=is_output_bias,
        verbose=verbose,
        is_return_sequences=is_return_sequences)(inputs)

    # must compute loss and then backward!

    #for output_tensor in outputs:
    #    print(output_tensor.data)

    # for tuple of output tensor, concat first, and compute loss!!!
    y_preds = outputs[:n_time_step]
    concat_y_pred, = Concat()(*y_preds)
    concat_y, = Concat()(*targets)
    loss, = MSELoss()(concat_y_pred, concat_y)
    print(loss.data)
    loss.backward()
    print(inputs[0].grad.data.shape)


if __name__ == '__main__':
    test_rnn()