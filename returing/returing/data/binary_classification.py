import numpy as np
np.random.seed(20170430)

def generate_data():

    """
    att1,att2,att3,attr3,y
    y = 0 or 1
    """
    input_path = '/Users/flyingman/Data/bi_iris'

    X = []
    Y = []

    for line in open(input_path, 'r'):
        buf = line[:-1].split(",")
        x = buf[:-1]
        y = buf[-1]
        X.append(x)
        Y.append([y])

    return np.array(X).astype(np.float), np.array(Y).astype(np.float)