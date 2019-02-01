from returing.nn.module.module import Module
from returing.nn.module.rnn.rnn import RNN
from returing.nn.tensor.tensor import Tensor
from returing.nn.function.base.concat import Concat
import numpy as np
np.random.seed(20170430)


class BiDirectional(Module):

    # list of modules
    modules = None

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
                 verbose=1,
                 initializer=None,
                 is_stateful=True,
                 ):
        super(BiDirectional, self).__init__()

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
        self.initializer = initializer
        self.verbose = verbose

        self.rnn_module1 = RNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_time_step=self.n_time_step,
            batch_size=self.batch_size,
            is_output_bias=self.is_output_bias,
            is_hidden_bias=self.is_hidden_bias,
            is_return_sequences=self.is_return_sequences,
            is_stateful=self.is_stateful,
            verbose=self.verbose,
            initializer=self.initializer
        )

        self.rnn_module2 = RNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_time_step=self.n_time_step,
            batch_size=self.batch_size,
            is_output_bias=self.is_output_bias,
            is_hidden_bias=self.is_hidden_bias,
            is_return_sequences=self.is_return_sequences,
            is_stateful=self.is_stateful,
            verbose=self.verbose,
            initializer=self.initializer
        )

        self.modules = [self.rnn_module1, self.rnn_module2]

    def forward(self, inputs):
        # inputs: tuple of input tensor, contain raw input data.
        if len(inputs) == 1:
            inputs, = inputs
            n_samples = inputs[0].data.shape[0]

            # initial state, be initialized or passed.
            fw_state = Tensor(np.random.randn(
                n_samples, self.n_time_step, self.hidden_dim))
            bw_state = Tensor(np.random.randn(
                n_samples, self.n_time_step, self.hidden_dim))

        elif len(inputs) == 3:
            inputs, fw_state, bw_state = inputs

        fw_inputs = tuple(list(inputs) + [fw_state])

        inputs.reverse()
        bw_inputs = tuple(list(inputs) + [bw_state])

        fw_outputs, fw_state = self.rnn_module1(fw_inputs)
        bw_outputs, bw_state = self.rnn_module2(bw_inputs)

        outputs = Concat()(fw_outputs, bw_outputs)

        return outputs, fw_state, bw_state
