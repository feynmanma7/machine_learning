import numpy as np
np.random.seed(20170430)

def relu(x):
    return x * (x > 0)


def relu_grad(x):
    return 1 * (x > 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_grad(y):
    return y * (1 - y)


def safe_read_dict(dictory, key, default=0):
    return dictory[key] if key in dictory else default


def get_shape_by_coord_tuple(coordinate_tuple):
    """
    Example Input: ((0, 2), (1, 3), (1, 4))
    Example Output: (2, 2, 3)
    """
    shape = []
    for coord in coordinate_tuple:
        shape.append(coord[1] - coord[0])
    return tuple(shape)


def set_sub_ndarray(A, B, coordinate_tuple, is_add=False):
    assert isinstance(coordinate_tuple, tuple)
    """
    # Input:
        A: ndarray
        B: ndarray
        coordinate_tuple: (coord_1, coord_2, ..., coord_N)
            coord_i: 2D-ndarray, [start_idx, end_idx]
            
    # Output:
        A Sub-ndarray of A.
    """
    shape = []
    slice_tuple = []
    for coordinate in coordinate_tuple:
        assert isinstance(coordinate, tuple)
        assert len(coordinate) == 2
        slice_tuple.append(
            slice(coordinate[0], coordinate[1]) # slice object
        )
        shape.append(coordinate[1] - coordinate[0])

    # assert B.size == coordinate_tuple.size(To Be Computed)

    # Note: B.data must be reshaped before used.
    if is_add:
        # Add to the raw data
        A[tuple(slice_tuple)] += B.reshape(tuple(shape))
    else:
        # Replace the raw data
        A[tuple(slice_tuple)] = B.reshape(tuple(shape))

    return A


def get_sub_ndarray(A, coordinate_tuple):
    assert isinstance(coordinate_tuple, tuple)

    slice_tuple = []
    for coord in coordinate_tuple:
        assert isinstance(coord, tuple)
        assert len(coord) == 2
        slice_tuple.append(slice(coord[0], coord[1]))

    return A[tuple(slice_tuple)]

    

