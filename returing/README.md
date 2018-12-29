<h1>Returing Deep Learning Framework</h1>

+ Project Address

> returing.returing.nn

Returing autograd-supported deep learning framework is a simple framework to build deep learning cores from scratch.

There're four core elements of this framework.

+ <b>Tensor</b>
+ <b>Operation</b>
+ Model(Operation)
+ Optimizer(Tensor)

The <b>Tensor</b> and <b>Operation</b> Part are the two Building Blocks of the computational graph which is actually a tensor-flow(operation) graph.

# Tensor
Base building block.
The object with data, including the Input data and data Obtained
from some specific operation.

# Operation
The way to process the data,
including forward and backward propagation.

# Model(Operation)
Abstract package of tensors and operations.

# Optimizer(Tensor)
The way to update parameters, let the project approximate our target
as much as possible.


