from returing.nn.module.module import Module
from returing.nn.function.base.repeat import Repeat
from returing.nn.function.base.add_fn import BatchAdd, Add
from returing.nn.function.base.reshape import Reshape
from returing.nn.function.base.dot_fn import Dot
from returing.nn.function.activation.tanh import Tanh

import numpy as np
np.random.seed(20170430)


class GRUCell(Module):

    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 output_dim=None,
                 n_time_step=None,
                 batch_size=None,
                 is_output_bias=True,
                 is_hidden_bias=True,
                 is_return_sequences=True,
                 verbose=1
                 ):
        super(GRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_time_step = n_time_step
        self.batch_size = batch_size
        self.is_output_bias = is_output_bias
        self.is_hidden_bias = is_hidden_bias
        self.is_return_sequences = is_return_sequences
        self.verbose = verbose

    def forward(self, inputs):
        """
        # Input
        X_0, X_1, ..., X_{T-1}, h_0, W_hy, W_hh, W_xh, b_o, b_h
            X: [n_samples, n_time_step, input_dim]
            X_t: [n_samples, input_dim]
            h_0: initial_state, must be passed.

        # Output
        if return_sequences:
            return y_0, y_1, ..., y_{T-1}, h_{T-1}
        else:
            return y_{T-1}, h_{T-1}

        y_pred: [n_samples, n_time_step, output_dim]
        h_t: [n_samples, n_time_step, hidden_dim], t = 0, 1, ..., n_time_step-1


        Three Gates

            Use Two biases in one gate ?

        + `reset_gate`: r_g_t = sigmoid ( h_{t-1} \cdot W_hh_r + X_t \cdot W_xh_r + b_h_r )

        + `update_gate`: u_g_t = sigmoid ( h_{t-1} \cdot W_hh_u + X_t \cdot W_xh_u + b_h_u )

        + `new_gate`: n_g_t = tanh ( X_t \cdot W_xh_n + b_h_n
                + r_g_t * ( h_{t-1} \dot W_hh_n + b_h_n ) )

        + `output`: h_t = (1 - u_g_t) * n_g_t + u_g_t * h_{t-1}


        ## `y_t`
            W_{hy}: [hidden_dim, output_dim]
            h_t: [n_samples, n_time_step, hidden_dim]
            b_o: [output_dim, ]
        y_t = h_t.dot(W_{hy}) + b_o

        ## `h_t`
            W_{hh}: [hidden_dim, hidden_dim]
            h_{t-1}: [n_samples, n_time_step, hidden_dim]

            X_t: [n_samples, n_time_step, input_dim]
            W_{xh}: [input_dim, hidden_dim]
            b_h: [hidden_dim, ]

        h_t = h_{t-1}.dot(W_{hh}) + X_t.dot(W_{xh}) + b_h
        h_t = tanh(h_t)

        """
        # In the order of x --> h --> o
        #X_0, X_1, ... , X_{T-1}, h_0, W_xh, W_hh, W_hy, b_h, b_o = inputs

        inputs, = inputs

        T = self.n_time_step

        h_t = inputs[T]
        W_xh = inputs[T+1]
        W_hh = inputs[T+2]
        W_hy = inputs[T+3]
        b_h = inputs[T+4]
        b_o = inputs[T+5]

        #hiddens = []
        y_preds = []

        #n_samples = inputs[0].data.shape[0]

        if self.is_hidden_bias:
            # b_h: [hidden_dim, ]
            # expanded_b_h: [hidden_dim * n_time_step, ]
            expanded_b_h, = Repeat(repeat_times=self.n_time_step)(b_h)

            # expanded_b_h: [n_time_step * hidden_dim]
            expanded_b_h, = Reshape(target_shape=
                           (self.n_time_step, self.hidden_dim))(expanded_b_h)

        if self.is_output_bias:
            # b_o: [output_dim, ]
            # expanded_b_o: [output_dim * n_time_step, ]
            expanded_b_o, = Repeat(repeat_times=self.n_time_step)(b_o)

            # expanded_b_o: [n_time_step, output_dim]
            expanded_b_o, = Reshape(target_shape=
                           (self.n_time_step, self.output_dim))(expanded_b_o)

        for t in range(T):
            X_t = inputs[t]

            # `h_t`

            # X_t: [n_samples, n_time_step, input_dim]
            # W_xh: [input_dim, hidden_dim]
            # xh: [n_samples, n_time_step, hidden_dim]
            xh, = Dot()(X_t, W_xh)

            # h_t: [n_samples, n_time_step, hidden_dim], here h_t is actually h_{t-1}
            # W_hh: [hidden_dim, hidden_dim]
            # hh: [n_samples, n_time_step, hidden_dim]
            hh, = Dot()(h_t, W_hh)

            h_t, = Add()(xh, hh)

            if self.is_hidden_bias:
                # h_t: [n_samples, n_time_step, hidden_dim]
                # expanded_b_h: [n_time_step, hidden_dim]
                h_t, = BatchAdd()(h_t, expanded_b_h)

            # !!! h_t must be <b>Activated</b>!
            h_t, = Tanh()(h_t)

            #hiddens.append(h_t)

            # `y_t`

            # h_t: [n_samples, n_time_step, hidden_dim]
            # W_hy: [hidden_dim, output_dim]
            # y_t: [n_samples, n_time_step, output_dim]
            y_t, = Dot()(h_t, W_hy)

            if self.is_output_bias:
                # y_t: [n_samples, n_time_step, output_dim]
                # expanded_b_o: [n_time_step, output_dim]
                y_t, = BatchAdd()(y_t, expanded_b_o)

            if self.is_return_sequences:
                y_preds.append(y_t)

        if self.is_return_sequences:
            #outputs = y_preds + hiddens
            outputs = y_preds
        else:
            #outputs = y_preds[-1] + hiddens
            outputs = y_preds[-1]

        #outputs = outputs + hiddens
        outputs = outputs + [h_t]
        outputs = tuple(outputs)

        #return tuple(outputs), tuple(hiddens)
        return outputs
