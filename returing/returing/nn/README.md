<h1>ReTuring Deep Learning Framework</h1>

# Features

+ Auto-grad supported

+ Single CPU, Single-Thread currently, 
Multi-CPU(GPU), Multi-Thread in the future

+ PyTorch(Chainer)-like, way of computational graph

+ Functional Programming way, final goal, to be designed and developed 

# Building Blocks

+ Function

> `Node` of the computational graph.

+ Tensor

> `Edge` of the computational graph.

# Core Elements

## 0. Function

<b>Function</b> is the fundamental or atom function, 
which can be used directly to compose complex functions(module),
whose `forward` and `backward` method must be implemented. 

+ Attrs

> `inputs`: tuple of input tensor.

> `outputs`: tuple of output tensor.

+ Methods

> `forward(inputs)`: `inputs` is tuple of input tensor, the function
return tuple of output tensor computes as the function itself. 
Some tensors are retained in the cache if necessary. 
Must be implemented in all of the subclasses. 
`forward` is called in the `__call__` function of the instance, 
which means once a `Function` or `Module` object is <b>instantiated</b>, 
the method is called immediately. Tuple of output tensor is returned.

> `backward(grads)`: `grads` is tuple of tensor which is
 gradient of the corresponding output tensor.
 
> `__call__`: Call the `forward` function, 
and trace the relationships to build the computational graph physically or logically.   

## 1. Tensor

+ Attrs

> `data`: numpy.ndarray (or like).

> `requires_grad`: bool, default False, 
if True, `grad` is computed.

> `grad`: tensor, if `requires_grad`, `grad` is computed, 
but only retained if current tensor is leaf node or set to be retained explicitly.

> `grad_fn`: `Function` (or `Module`) or None, if not none, is the 
function that creates current tensor.

> `is_leaf`: bool, default False. The computational graph
is built from leaf to root in the forward way, 
while in the backward procedure, 
the error is propagated from root recursively until meet the leaf node.

+ Methods

> `backward()`: Pass current tensor itself and its `grad` to the `backward` 
function of current tensor's `grad_fn`, 
and <b>recursively</b> call `backward` of the `*inputs` tensors of `grad_fn`. 

## 2. Module(object)

Composite of `Function`, only `forward` function is need to be implemented
if only existing `Function` used.

Some neural network like Convolutional Neural Network 
or Full Connected Layer composed of input tensors and
weights (defined as `Parameter`).

A `Module` can be seen as one `Function` logically, 
with input tensors and output tensors as `Edge`,
and a function as `Node`. 

It's better to know what you're implemented is a `Function` or 
a `Module`. 

## 3. Parameter(Tensor)

`Tensor` need to be initialized and updated iteratively,
`requires_grad` default.  

## 4. Optimizer(Function)

Rule to update `Parameter`, is a `Function` logically, 
with input tensors and output tensors and a function (update rule).

## 5. Initializer(Function)

Specifically an optimizer.

## 6. Model(object)

Aggregation of modules(or functions), 
thus parameters(optional) can be obtained for 
`Optimizer` to update.



 



