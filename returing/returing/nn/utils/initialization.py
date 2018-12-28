from returing.nn.tensor import Tensor

import numpy as np
np.random.seed(20170430)


def random_init_tensor(shape, **kwargs):
    # Valid numpy shape, int or tuple of int
    # --- assert isinstance(shape, int) or isinstance(shape, tuple)

    return Tensor(np.random.random(shape), **kwargs)