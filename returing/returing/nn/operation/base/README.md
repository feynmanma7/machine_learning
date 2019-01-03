<h1>Operation</h1>

# Backward (grad_out)
+ Set cur_grad according to current operation.
+ grad_out is gradient from backwards, maybe empty. 
If not empty, has the same shape with Y_pred.  

> + If not total grad, total_grad = cur_grad * grad_out

> + Else total_grad += cur_grad * grad_out

# Convolutional Neural Network
## Conv2D

+ Example Usage
> Y_pred = Conv2D()(X)

+ Input

X: [n_samples, n_input_channel, input_width, input_height]

+ Input parameters

> K: Kernel_size
> P: Padding
> S: Stride

+ Output

Y_pred: [n_samples, n_output_channel, output_width, output_height]



