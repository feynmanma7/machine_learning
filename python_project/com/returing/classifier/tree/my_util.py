import numpy as np
np.random.seed(20170430)

def generate_data():

    attr_1_set = np.array([1, 2, 3, 4])
    attr_2_set = np.array([5, 6, 7])
    attr_3_set = np.array([8, 9])

    y_set = np.array([0, 1])

    num_sample = 100

    X = []
    Y = []
    for _ in range(num_sample):
        attr_1 = np.random.choice(attr_1_set)
        attr_2 = np.random.choice(attr_2_set)
        attr_3 = np.random.choice(attr_3_set)
        y = np.random.choice(y_set)

        X.append([attr_1, attr_2, attr_3])
        Y.append([y])

    return np.array(X), np.array(Y)