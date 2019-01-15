from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Linear(Function):

    inputs = None
    outputs = None

    def __init__(self,
                 n_in_features,
                 n_out_features,
                 is_bias=True):
        super(Linear, self).__init__()

        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.is_bias = is_bias

    def forward(self, inputs):
        """
        Input: X, W, bias; tuple of tensor
            X: [n_samples, n_in_features]
            W: [n_in_features, n_out_features]
            bias: [n_output_features, ]

        Output: y_pred, tensor
            y_pred: [n_samples, n_out_features]

        Use broadcast of numpy when add bias
        y_pred = dot(X, W) + bias (if needed)
        """
        X, W, bias = inputs

        y_pred_data = np.dot(X.data, W.data)

        if self.is_bias:
            y_pred_data += bias.data

        y_pred = Tensor(y_pred_data)

        n_samples = X.data.shape[0]

        bias_requires_grad = bias.requires_grad if isinstance(bias, Tensor) else False

        self.saved_context = X.requires_grad, W.requires_grad, bias_requires_grad, \
                             n_samples, X, W

        return y_pred,

    def backward(self, grads):
        X_requires_grad, W_requires_grad, bias_requires_grad, n_samples, X, W = \
            self.saved_context

        X_grad = None
        W_grad = None
        bias_grad = None

        if X_requires_grad:
            # W.data: [n_in_feature, n_out_features]
            # X.data: [n_samples, n_in_features]
            X_grad_data = np.repeat(np.sum(W.data, axis=1), n_samples).reshape(
                (n_samples, self.n_in_features))

            if isinstance(grads, tuple):
                y_pred_grad, = grads
                # y_pred_grad_data: [n_samples, n_out_features]
                X_grad_data *= np.repeat(np.sum(y_pred_grad.data, axis=1), self.n_in_features).\
                    reshape((n_samples, self.n_in_features))

            X_grad = Tensor(X_grad_data)

        if W_requires_grad:
            # X.data: [n_samples, n_in_features]
            # W.data: [n_in_features, n_out_features]
            W_grad_data = np.repeat(np.sum(X.data, axis=0), self.n_out_features).reshape(
                (self.n_in_features, self.n_out_features))

            if isinstance(grads, tuple):
                y_pred_grad, = grads
                # y_pred_grad.data: [n_samples, n_out_features]
                W_grad_data *= np.sum(y_pred_grad.data, axis=0).reshape((1, self.n_out_features))

            W_grad = Tensor(W_grad_data)

        if bias_requires_grad:
            # bias_grad_data: [n_out_features, ]
            bias_grad_data = np.ones((self.n_out_features, ))

            if isinstance(y_pred_grad, Tensor):
                # y_pred_grad.data: [n_samples, n_out_features]
                bias_grad_data *= np.sum(y_pred_grad.data, axis=0)

            bias_grad = Tensor(bias_grad_data)

        #self.saved_context = None

        return X_grad, W_grad, bias_grad