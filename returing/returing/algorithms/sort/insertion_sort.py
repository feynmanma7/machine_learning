from returing.algorithms.sort.sort_util \
    import less, exchange, generate_data


def insertion_sort(arr):
    for i in range(len(arr)-1):
        for j in range(i+1, 0, -1):
            if less(arr, j, j-1):
                exchange(arr, j, j-1)
            else:
                break

if __name__ == '__main__':
    arr = generate_data()

    print(arr)
    insertion_sort(arr)
    print(arr)


