from returing.nn.tensor.tensor import Tensor
from returing.nn.module.rnn import rnn_layer
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
    for _ in range(n_time_step):
        input_tensor = Tensor(np.random.rand(n_samples, n_time_step, input_dim),
                              requires_grad=True,
                              is_leaf=True)
        inputs.append(input_tensor)
    inputs = tuple(inputs)

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
    for output_tensor in outputs:
        print(output_tensor.data)


if __name__ == '__main__':
    test_rnn()