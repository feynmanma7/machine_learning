import numpy as np
np.random.seed(20170430)

from matplotlib import pyplot as plt


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

def _compute_area(X, Y):
    # X: numpy ndarray: n_samples * 1
    # Y: numpy ndarray: n_samples * 1

    assert X.shape == Y.shape
    assert X.shape[1] == 1

    n_samples = X.shape[0]

    sorted_idx = np.argsort(X[:, 0])
    sorted_X = X[sorted_idx]
    sorted_Y = Y[sorted_idx]

    area = 0.
    for i in range(n_samples - 1):
        dx = sorted_X[i+1] - sorted_X[i]
        # area += dx * sorted_Y[i]
        # area += dx * sorted_Y[i+1]
        area += dx * (sorted_Y[i] + sorted_Y[i+1]) / 2

    return area

def compute_auc(Y, Y_pred, threshold=.5):
    assert Y.shape == Y_pred.shape

    sorted_idx = np.argsort(Y_pred[:, 0])

    sorted_Y = Y[sorted_idx]

    # fpr =  fp / (fp + tn ) , tpr = tp / (tp + fn)

    fpr = []
    tpr = []

    n_pos = np.sum(Y)
    n_pos_left = 0
    n_samples = Y.shape[0]


    for i in range(-1, n_samples):
        if i >= 0 and i < n_samples and sorted_Y[i] == 1:
            n_pos_left += 1

        n_left = sorted_Y[:i+1].shape[0]
        n_right = n_samples - n_left
        n_pos_right = n_pos - n_pos_left

        fn = n_pos_left
        tn = n_left - fn

        tp = n_pos_right
        fp = n_right - tp

        cur_fpr = fp * 1. / (fp + tn)
        cur_tpr = tp * 1. / (tp + fn)
        fpr.append(cur_fpr)
        tpr.append(cur_tpr)

    fpr = np.array(fpr).astype(np.float).reshape((len(fpr), 1))
    tpr = np.array(tpr).astype(np.float).reshape((len(tpr), 1))

    auc = _compute_area(fpr, tpr)

    return auc



def test_compute_auc():

    Y = np.array([1, 0, 1, 0, 1,    0, 1, 0, 1, 0]).reshape((10, 1))
    Y_pred = np.array([.9, 0.1, 0.8, 0.6, 0.4,    0.35, 0.85, 0.43, 0.65, 0.24 ]).reshape((10, 1))

    auc = compute_auc(Y, Y_pred)

    print("AUC = %.4f" % auc)

    """
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.show()
    """

if __name__ == '__main__':
    test_compute_auc()

    a = [0, 0.8, 0.6, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    b = [0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0]

    # plt.plot(a, b)
    # plt.show()