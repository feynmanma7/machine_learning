<h1>Operation</h1>

Operation is atom or composite.

# 0. Components
## Atom Operation
Only support <b>unary</b> and <b>binary</b> operation.

### Forward
The forward function <b>must</b> be implemented.
There are current four elements in the checklist.

+ Save input tensor for current operation.

+ Point self as the grad_fn of return tensor, 
set the `requires_grad` attribute according to the input tensor.

+ Point the parent-child relationships.

+ Return a new tensor.

### Backward
The backward function <b>must</b> be implemented. 

Update the grad of the input tensor(s) of current operation.

The grad must have the same shape with the corresponding tensor(s).

## Composite Operation
### Forward
The same implement with the atom operation. 

# 1. Implementations
## Unary Operation
### Math formula
+ Sum: np.sum(), ndarray to scalar
+ Pow:
+ Exp:
+ Log:

###  Activation function
+ ReLU
+ Sigmoid

## Binary Operation
### Math formula
+ Add: A + B
+ Subtract: A - B
+ Matmul: np.dot(A, B)

### Loss function
+ MSE
+ Cross_entropy
+ KL_divergence



class Operation(object):

    params = None

    def __init__(self, *args, **kwargs):
        super(Operation, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        pass