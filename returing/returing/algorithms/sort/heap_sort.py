from returing.algorithms.sort.sort_util \
    import less, exchange, generate_data

def build_heap(arr):
    for i in range(int(len(arr)/2)-1, -1, -1):
        heapify(arr, i, len(arr))
        #print(arr[i], ',', arr)

def get_left(i):
    return 2 * i + 1

def get_right(i):
    return 2 * i + 2

def get_parent(i):
    return int((i - 1) / 2)


def heapify(arr, idx, N):
    if idx < 0:
        return
    left_idx = get_left(idx)
    right_idx = get_right(idx)

    min_idx = -1
    if left_idx < N:
        min_idx = left_idx
    if right_idx < N:
        min_idx = left_idx if less(arr, left_idx, right_idx) else right_idx

    if min_idx == -1:
        return

    if less(arr, min_idx, idx):
        exchange(arr, min_idx, idx)
        heapify(arr, min_idx, N)

    #parent_idx = get_parent(idx)
    #heapify(arr, parent_idx)


def heap_sort(arr, end):

    build_heap(arr)
    print(arr)

    j = end - 1
    for i in range(len(arr) - 1):
        exchange(arr, 0, j)
        heapify(arr, 0, j)
        j -= 1


if __name__ == '__main__':
    arr = generate_data()

    print(arr)

    #build_heap(arr)
    #print(arr)

    heap_sort(arr, len(arr)) # [start, end)
    print(arr)