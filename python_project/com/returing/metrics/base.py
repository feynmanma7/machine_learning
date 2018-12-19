import numpy as np
np.random.seed(20170430)

def compute_accuracy(Y, Y_pred, threshold=0.5):
    assert Y.shape == Y_pred.shape
    n_correct = 0
    n_samples = Y.shape[0]
    for i in range(n_samples):
        y = 1 if Y_pred[i] >= threshold else 0
        if Y[i] == y:
            n_correct += 1

    accuracy = n_correct / n_samples
    return accuracy