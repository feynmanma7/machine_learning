from com.returing.classifier.tree.RegressionTree import RegressionTree, generate_data
from com.returing.metrics.base import compute_accuracy, compute_auc, compute_auc_math
from sklearn.metrics import roc_auc_score
import numpy as np
np.random.seed(20170430)


class GBDT:

    max_n_trees = 5
    trees = None

    def __init__(self,
                 max_n_trees = 5):
        self.max_n_trees = max_n_trees
        self.trees = []

    def fit(self, X, Y):

        Y_t = Y.copy()

        for _ in range(self.max_n_trees):
            rt = RegressionTree()
            rt.fit(X, Y_t)
            self.trees.append(rt)

            Y_pred = rt.predict(X)

            # Negative gradient
            Y_t = Y_t - Y_pred


    def predict(self, X):

        results = []

        for x in X:
            result = 0.
            for rt in self.trees:
                result += rt._predict(x)

            results.append(result)

        return np.array(results).astype(np.float).reshape((X.shape[0], 1))



def main():
    X, Y = generate_data()

    clf = GBDT(max_n_trees=5)
    clf.fit(X, Y)

    Y_pred = clf.predict(X)

    for y_pred, y in zip(Y_pred, Y):
        print(y_pred, y)

    accuracy = compute_accuracy(Y, Y_pred)
    auc = compute_auc(Y, Y_pred)
    auc_math = compute_auc_math(Y, Y_pred)

    print("Train accuracy=%.4f" % accuracy)
    print("Train auc=%.4f" % auc)
    print("Train auc_math=%.4f" % auc_math)

    auc = roc_auc_score(Y, Y_pred)
    print("sklearn auc=%.4f" % auc)


if __name__ == '__main__':
    main()