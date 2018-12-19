from RegressionTree import RegressionTree, generate_data
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

    clf = GBDT()
    clf.fit(X, Y)

    results = clf.predict(X)

    for result, y in zip(results, Y):
        print(result, y)


if __name__ == '__main__':
    main()