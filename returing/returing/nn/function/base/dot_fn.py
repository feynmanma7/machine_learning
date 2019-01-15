from returing.nn.function.function import Function
from returing.nn.tensor.tensor import Tensor
import numpy as np


class Dot(Function):

    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, inputs):
        """
        #Note: Current only can be used for
        #    a 3-dim Tensor `A` Dot with
        #    a 2-dim Tensor `B`.

        Input
            a: [shape_1, dim]
            b: [dim, shape_2]

        Output
            c = np.dot(a, b) [shape_1, shape_2]
        """
        A, B = inputs

        y_pred_data = np.dot(A.data, B.data)
        y_pred = Tensor(y_pred_data)

        self.saved_context = A, B

        return y_pred,

    def backward(self, grads):
        """
        C = AB
        A: [shape_1, dim]
        B: [dim, shape_2]

        A_grad = (B.T [shape_2, dim] aligns to [shape_1, dim])

        B_grad = (A.T [dim, shape_1] aligns to [dim, shape_2])
        """

        A, B = self.saved_context

        A_grad = None
        B_grad = None

        if A.requires_grad:
            # B.data: [dim, shape_2], B.data.T: [shape_2, dim]
            # A_grad_data: [shape_1, dim]
            # [shape_1, dim] * [dim, ] --> [shape_1, dim]
            A_grad_data = np.ones(A.data.shape) \
                          * np.mean(B.data.T, axis=0)

            if isinstance(grads, tuple):
                y_pred_grad, = grads
                if isinstance(y_pred_grad, Tensor):
                    # y_pred_grad.data: [shape_1, shape_2]
                    # A_grad_data: [shape_1, dim]

                    #A_dim = len(A.data.shape)
                    B_dim = len(B.data.shape)
                    C_dim = len(y_pred_grad.data.shape)

                    # mean_y_pred_data: [shape_1]
                    mean_y_pred_data = np.mean(y_pred_grad.data, axis=
                        tuple(range(C_dim - B_dim + 1, C_dim)))

                    # mean_y_pred_data: bb [shape_1]
                    # A_grad_data: aa [shape_1, dim]
                    # (aa.T * bb.T).T
                    A_grad_data = (A_grad_data.T * mean_y_pred_data.T).T

            A_grad = Tensor(A_grad_data)


        if B.requires_grad:
            # B_grad_data: [dim, shape_2]
            # A.data: [shape_1, dim]

            A_dim = len(A.data.shape)
            #B_dim = len(B.data.shape)

            B_grad_data = (np.ones(B.data.shape).T * \
                          np.mean(A.data, axis=tuple(range(0, A_dim-1)))).T

            if isinstance(grads, tuple):
                y_pred_grad, = grads
                if isinstance(y_pred_grad, Tensor):
                    # y_pred_grad.data: [shape_1, shape_2]
                    # B_grad_data: [dim, shape_2]

                    # [shape_2]
                    mean_y_pred_data = np.mean(y_pred_grad,
                                               axis=tuple(range(0, A_dim-1)))
                    B_grad_data *= mean_y_pred_data

            B_grad = Tensor(B_grad_data)

        return A_grad, B_grad