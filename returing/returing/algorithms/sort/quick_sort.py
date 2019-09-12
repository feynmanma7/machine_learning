from returing.algorithms.sort.sort_util \
    import less, exchange, generate_data


def partition(arr, start, end):
    # [start, end)
    #pivot = arr[start]

    i = start + 1
    j = end - 1

    while True:

        while True:
            # Find the first one bigger than arr[start]
            if i < end and less(arr, i, start):
                i += 1
            else:
                break

        while True:
            # Find the first one smaller than arr[start]
            if j > start and less(arr, start, j):
                j -= 1
            else:
                break

        if i >= j:
            break

        exchange(arr, i, j)
        # i += 1
        # j -= 1
        
        #print(pivot, arr[start:end])

    exchange(arr, j, start)
    #print(pivot, arr[start:end])

    return j


def quick_sort(arr, start, end):
    # [start, end)

    if start >= end:
        return

    pivot_idx = partition(arr, start, end) # [start, end)
    quick_sort(arr, start, pivot_idx) # [start, pivot_idx)
    quick_sort(arr, pivot_idx+1, end) # [pivot_idx+1, end)


if __name__ == '__main__':
    arr = generate_data()

    print(arr)
    quick_sort(arr, 0, len(arr))
    print(arr)