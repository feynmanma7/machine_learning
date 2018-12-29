"""
Operation is atom.
Only support unary and binary operation.

# ====== Unary Operation
## === Math formula
### Sum: np.sum(), ndarray to scalar
### Pow:
### Exp:
### Log:

## === Activation function
### ReLU
### Sigmoid

# ====== Binary Operation
## === Math formula
## Add: A + B
## Subtract: A - B
## Matmul: np.dot(A, B)

## === Loss function
### MSE
### Cross_entropy
### KL_divergence

"""


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