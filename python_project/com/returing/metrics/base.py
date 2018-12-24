import numpy as np
np.random.seed(20170430)
from sklearn.metrics import roc_auc_score

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

    # sorted_idx = np.argsort(X[:, 0])
    # sorted_X = X[sorted_idx]
    # sorted_Y = Y[sorted_idx]

    area = 0.
    for i in range(n_samples - 1):
        #dx = sorted_X[i+1] - sorted_X[i]
        dx = X[i+1] - X[i]
        # area += dx * sorted_Y[i]
        # area += dx * sorted_Y[i+1]
        #area += dx * (sorted_Y[i] + sorted_Y[i+1]) / 2
        area += dx * (Y[i] + Y[i+1]) / 2

    return np.asscalar(area)


def compute_auc_math(Y, Y_pred, pos_label=1):
    assert Y.shape == Y_pred.shape
    assert Y.shape[1] == 1

    sorted_idx = np.argsort(Y_pred[:, 0])
    sorted_Y = Y[sorted_idx]

    n_samples = Y.shape[0]
    n_pos = np.sum(Y) # M
    n_neg = n_samples - n_pos # N

    assert n_pos > 0
    assert n_neg > 0

    auc = 0.

    for i in range(n_samples):
        if sorted_Y[i] == pos_label:
            rank = i + 1
            auc += rank

    auc = (auc - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def compute_auc(Y, Y_pred, pos_label=1):
    assert Y.shape == Y_pred.shape
    assert Y.shape[1] == 1

    # From high to low, Assume left is pos, right is neg
    sorted_idx = np.argsort(-Y_pred[:, 0])
    sorted_Y = Y[sorted_idx]

    # fpr =  fp / (fp + tn ) , tpr = tp / (tp + fn)

    fpr = []
    tpr = []

    n_pos = np.sum(Y)
    n_pos_left = 0
    n_samples = Y.shape[0]

    # fpr.append(0.)
    # tpr.append(0.)

    for i in range(n_samples):
        # if i >= 0 and i < n_samples and sorted_Y[i] == pos_label:
        if sorted_Y[i] == pos_label:
            n_pos_left += 1

        n_left = i+1
        n_right = n_samples - n_left
        n_pos_right = n_pos - n_pos_left

        tp = n_pos_left
        fp = n_left - tp

        fn = n_pos_right
        tn = n_right - fn

        cur_fpr = fp * 1. / (fp + tn)
        cur_tpr = tp * 1. / (tp + fn)
        fpr.append(cur_fpr)
        tpr.append(cur_tpr)

    #fpr.append(1.)
    #tpr.append(1.)

    fpr = np.array(fpr).astype(np.float).reshape((len(fpr), 1))
    tpr = np.array(tpr).astype(np.float).reshape((len(tpr), 1))

    auc = _compute_area(fpr, tpr)

    return auc



def test_compute_auc():

    Y = np.array([1, 0, 1, 0, 1,    0, 1, 0, 1, 0]).reshape((10, 1))
    Y_pred = np.array([.9, 0.1, 0.8, 0.6, 0.4,    0.35, 0.85, 0.43, 0.65, 0.24 ]).reshape((10, 1))

    auc = compute_auc(Y, Y_pred)
    auc_math = compute_auc(Y, Y_pred)
    auc_sk = roc_auc_score(Y, Y_pred)

    print("auc = %.4f" % auc)
    print("auc_math= %.4f" % auc_math)
    print("auc_sklearn = %.4f" % auc_sk)

    """
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.show()
    """

if __name__ == '__main__':
    # test_compute_auc()

    a = [0, 0.8, 0.6, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    b = [0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0]

    Y = np.array([0, 0, 1, 1]).reshape((4, 1))
    Y_pred = np.array([0.1, 0.4, 0.35, 0.8]).reshape((4, 1))

    auc = compute_auc(Y, Y_pred)
    auc_math = compute_auc_math(Y, Y_pred)
    auc_sk = roc_auc_score(Y, Y_pred)

    print(auc, auc_math, auc_sk)

    # plt.plot(a, b)
    # plt.show()