<h1>Attention is all you need</h1>

# Attention 

In the Encoder-Decoder Framework.

## Inputs

+ `hiddens`, tuple of hidden states of RNN(LSTM or GRU or Bi-directional Layer) Module. 

## Outputs

### Decoder

> `output`: y_t = g(y_{t-1}, s_t, c_t)

> `hidden`: s_t = f(s_{t-1}, y_{t-1}, c_t)

> `context`: c_t = \sum_{j=1}^{T_x} \alpha_{t, j} h_j, 
T_x is the length of current sequence.

> `weight`: \alpha_{t, j} = \frac{ \exp{e_{t, j}} } {\sum_{k=1}^{T_x} \exp{e_{t, k}} }

> `alignment model`: e_{t, j} = a(s_{t-1}, h_j) 

### Encoder 

RNN-based model to generate hidden states h_1, h_2, ..., h_T.

## Learnable Parameters

# Transformer

# BERT