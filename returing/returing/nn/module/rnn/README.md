<h1>Recurrent Neural Network</h1>

# Core Codes 

`*.py` and `*_cell.py`.

For example, 

> `rnn`: Module with parameters, call `rnn_cell`.

> `rnn_cell`: Basic RNN operation.

# RNN

## Inputs

+ `inputs`: 3D tensor [batch_size, timesteps, input_dim]

## Outputs

+ `outputs`:  

> If `is_return_sequences`, outputs is 3D tensor [batch_size, timesteps, output_dim], 
else outputs is 2D tensor [batch_size, output_dim].

> If `is_return_state`, return last hidden state, outputs is two tensor, 
(outputs, h_T).

## Learnable Parameters

Weights and bias(optional) from `inputs` to `hidden`, 
and from `hidden` to `hidden`. 

# RNNCell

## Inputs

+ `inputs`: tuple of tensor, (inputs, state, W_xh, W_hh, bias_h)

## Outputs

+ `outputs`: 

> If `is_return_sequences`, outputs is 3D tensor [batch_size, timesteps, output_dim], 
else outputs is 2D tensor [batch_size, output_dim].

> If `is_return_state`, return last hidden state, outputs is two tensor, 
(outputs, h_T).


# LSTM
