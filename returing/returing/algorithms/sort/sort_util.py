import random
random.seed(20170430)

def less(arr, i, j):
    return True if arr[i] < arr[j] else False

def exchange(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def generate_data():
    arr = list(range(10))
    random.shuffle(arr)

    return arr