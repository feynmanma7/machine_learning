from returing.nn.module.module import Module
from returing.nn.module.rnn import rnn_cell
from returing.nn.tensor.parameter import Parameter
from returing.nn.util.initialization import random_init
#from returing.nn.function.activation import sigmoid, tanh
from returing.nn.tensor.tensor import Tensor
import numpy as np
np.random.seed(20170430)


class RNN(Module):

    # list of Parameter
    parameters = None

    # list of Parameter
    modules = None

    # Parameter, [hidden_dim, hidden_dim]
    W_hh = None

    # Parameter, [hidden_dim, output_dim]
    W_hy = None

    # Parameter, [input_dim, hidden_dim]
    W_xh = None

    # Parameter, [output_dim, ]
    bias_o = None

    # Parameter, [hidden_dim, ]
    bias_h = None

    # Tensor, [n_samples, n_time_step, hidden_dim]
    #   If `is_stateful` is True,
    #   use final state of last batch as initial state of next batch.
    #state = None

    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 output_dim=None,
                 n_time_step=None,
                 batch_size=None,
                 is_output_bias=True,
                 is_hidden_bias=True,
                 is_bi_direction=True,
                 is_return_sequences=True,
                 is_stateful=True,
                 verbose=1,
                 initializer=None,
                 activation=None
                 ):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_time_step = n_time_step
        self.batch_size = batch_size
        self.is_output_bias = is_output_bias
        self.is_hidden_bias = is_hidden_bias
        self.is_bi_direction = is_bi_direction
        self.is_return_sequences = is_return_sequences
        self.is_stateful = is_stateful
        self.verbose = verbose
        self.initializer = initializer
        self.activation = activation

        self.rnn_cell_func = rnn_cell.RNNCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_time_step=self.n_time_step,
            batch_size=self.batch_size,
            is_output_bias=self.is_output_bias,
            is_hidden_bias=self.is_hidden_bias,
            verbose=self.verbose,
            is_return_sequences=self.is_return_sequences)

        # Initialize State, h_0: [hidden_dim, hidden_dim]
        #self.state = Tensor(np.random.randn(self.hidden_dim, self.hidden_dim))

        # Initialize Parameters
        if not initializer:
            initializer = random_init

        self.parameters = []

        # Parameter, [hidden_dim, hidden_dim]
        self.W_hh = Parameter(
            name='W_hh',
            data=initializer((self.hidden_dim, self.hidden_dim))
        )
        self.parameters.append(self.W_hh)

        # Parameter, [hidden_dim, output_dim]
        self.W_hy = Parameter(
            name='W_hy',
            data=initializer((self.hidden_dim, self.output_dim))
        )
        self.parameters.append(self.W_hy)

        # Parameter, [input_dim, hidden_dim]
        self.W_xh = Parameter(
            name='W_xh',
            data=initializer((self.input_dim, self.hidden_dim))
        )
        self.parameters.append(self.W_xh)

        # Parameter, [output_dim, ]
        if self.is_output_bias:
            self.bias_o = Parameter(
                name='bias_o',
                data=initializer((self.output_dim, ))
            )
            self.parameters.append(self.bias_o)

        # Parameter, [hidden_dim, ]
        if self.is_hidden_bias:
            self.bias_h = Parameter(
                name='bias_h',
                data=initializer((self.hidden_dim, ))
            )
            self.parameters.append(self.bias_h)

    def forward(self, inputs):
        # inputs: tuple of input tensor, contain raw input data.

        if len(inputs) == 1:
            inputs, = inputs
            n_samples = inputs[0].data.shape[0]

            # initial state, be initialized or passed.
            #state = Tensor(np.random.randn(
                #n_samples, self.n_time_step, self.hidden_dim))
            state = Tensor(np.zeros((n_samples, self.n_time_step, self.hidden_dim)))

        elif len(inputs) == 2:
            inputs, state = inputs

        new_inputs = tuple(list(inputs) + \
                    [state, self.W_xh, self.W_hh, self.W_hy, self.bias_h, self.bias_o])

        return self.rnn_cell_func(new_inputs)