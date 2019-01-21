<h1>Recurrent Neural Network</h1>

# Core Codes 

`*.py` and `*_cell.py`.

For example, 

> `rnn`: Module with parameters, call `rnn_cell`.

> `rnn_cell`: Basic RNN operation.

# RNN

## Inputs

+ `inputs`: 3D tensor [n_samples, timesteps, input_dim]

## Outputs

+ `outputs`:  

> For each timestep, **GetSubTensor** of inputs to obtain `input_tensor`_t; 

> call `RNNCell`, obtain `output_tensor`_t;

> If `is_return_sequences`, 
outputs is 3D tensor [n_samples, timesteps, output_dim] by
**Concat** the tuple of output_tensors(hidden states),  
else outputs is 2D tensor [batch_size, output_dim].

> If `is_return_state`, return last hidden state, outputs is two tensor, 
(outputs, h_T).

## Learnable Parameters

Weights and bias(optional) from `inputs` to `hidden`, 
and from `hidden` to `hidden`. 

# RNNCell

## Inputs

+ `inputs`: tuple of tensor, (input_tensor, state, W_xh, W_hh, bias)

> `input_tensor`: X_t, tensor of shape (bactch_size, input_dim).

> `state`: h_{t-1}, hidden state before current time.

> `W_xh`: Parameters of weights from input to hidden layer.

> `W_hh`: Parameters of weights from hidden to hidden layer.

> `bias`: Parameter of bias(optional).

## Outputs

+ `outputs`: (hiddens, h_T) or (h_T, h_T), 
`hiddens` is 3D tensor [n_samples, timestep, hidden_dim], 
while `h_T` is 2D tensor [n_samples, hidden_dim].  

> h_t = W_xh \cdot X_t + W_hh \cdot h_{t-1} + bias

> output_tensor = activation(output_tensor) 

> If `is_return_sequences`, 
outputs is 3D tensor [n_samples, timesteps, output_dim], by
**Concat** the tuple of hidden states and **Reshape** to the proper shape;
else outputs is 2D tensor [n_samples, output_dim].

> If `is_return_state`, return last hidden state,
outputs is tuple of two tensor, 
(outputs, h_T).


# LSTM

## Inputs

+ `inputs`: input_tensor, [n_samples, timestep, input_dim]

## Outputs

> For each timestep, **GetSubTensor** to obtain a subtensor of current timestep, 
X_t; 

> Call LSTMCell, obtain h_t, c_t;

+ `outputs`: (hiddens, h_T, c_T) or (h_T, h_T, c_T), 
if not `is_returned_state`, the last two states is omitted.


## Learnable Parameters

+ `input_gate`: 

> W_ih_i, W_hh_i, b_i

+ `forget_gate`:

> W_ih_f, W_hh_f, b_f

+ `output_gate`:

> W_ih_o, W_hh_o, b_o

> `cell_gate`:

> W_ih_c, W_hh_c, b_c


# LSTMCell

## Inputs

+ `inputs`: tuple of tensor, (input_tensor, h_0, c_0, 
W_ih_i, W_hh_i, b_i,
W_ih_f, W_hh_f, b_f,
W_ih_o, W_hh_o, b_o, 
W_ih_c, W_hh_c, b_c), 
in the order of 
input_gate, forget_gate, output_gate, cell_gate. 
`input_tensor` is in the shape of [n_samples, input_dim].  

## Outputs

+ `outputs`: (hiddens, h_T, c_T) or (h_T, h_T, c_T), 
`hiddens` is 3D tensor [n_samples, timestep, hidden_dim], 
`h_T` is 2D tensor [n_samples, hidden_dim],
`c_T` is 2D tensor [n_samples, hidden_dim].

> `input_gate`: in_g_t = sigmoid( W_ih_i \cdot X_t + W_hh_i \cdot h_{t-1} + b_i ), **sigmoid**-like activation function is must, 
to let the output of gate is between 0 and 1.  

> `forgate_gate`: f_g_t = sigmoid( W_ih_f \cdot X_t + W_hh_f \cdot h_{t-1} + b_f )

> `output_gate`: out_g_t = sigmoid( W_ih_o \cdot X_t + W_hh_o \cdot h_{t-1} + b_o )

> `cell_gate`:  c_g_t = tanh( W_ih_c \cdot X_t + W_hh_c \cdot h_{t-1} + b_c ), 
**tanh** is used here, for `cell_gate` is the information gate not a control gate. 

> `cell`: c_t =  c_{t-1} * f_g_t + c_g_t * in_g_t, 
`*` is element-wise multiplication.

> `hidden`: h_t = tanh(c_t) * out_g_t

