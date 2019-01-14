<h1>Convolutional Neural network</h1>

# Core Codes

> conv2d_layer

> conv2d_module


# Input, Output & Parameters

+ Input

> X: [n_samples, n_input_channel, 
input_width, input_height], e.g. batch of images

+ Hyper-parameters

> stride: int, size of move around.

> padding: int, size of pixel to pad
around the raw input.

> kernel_size: int, symmetric for simplicity, 
size of convolution window.

> is_bias: bool, if true, add bias.

+ Output

> y_pred: [n_samples, n_output_channel, 
output_width, output_height], 
out_size = (in_size - kernel_size + 2*padding)/stride + 1.

+ Learnable Parameters

> W: [n_output_channel, n_input_channel, 
kernel_size, kernel_size]

> bias: [n_output_channel, ] (optional, default is_bias=True)

# Design

The Convolutional procedure is composed of some <b>atom</b> `Function`.

Details are in the file of `conv2d_module.py`.

## Convolutional Procedure

+ Padding the input.

+ Compute each atom result of the convolutional operation.

+ Concat all of the atom result, reshape to the target shape.

+ Add bias(repeated to the specific shape) if needed.

## Atom convolutional operation

+ Get sub-tensor of X w.r.t output_width_index 
and output_height_index. 

+ Get sub-tensor of W w.r.t ouput_channel.

+ BatchMatMul of sub_X and sub_W.

+ BatchSum the result of BatchMatMul.


## Atom Function

> `GetSubTensor`:

> `BatchMatMul`:

> `BatchSum`:

> `BatchAdd`:

> `Reshape`:

> `Repeat`:

> `Transpose`:

> `Concat`: 

> `Padding2D`:


