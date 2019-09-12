import math
from returing.algorithms.sort.sort_util \
    import exchange, generate_data


def find_min_index(arr, start, end):
    min_idx = -1
    min_value = math.inf

    for i in range(start, end):
        if arr[i] < min_value:
            min_value = arr[i]
            min_idx = i

    return min_idx


def selection_sort(arr):

    for i in range(len(arr)):
        min_idx = find_min_index(arr, i, len(arr))
        exchange(arr, i, min_idx)

    #return arr



if __name__ == '__main__':
    arr = generate_data()

    print(arr)
    selection_sort(arr)
    print(arr)
