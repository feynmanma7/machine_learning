#encoding:utf-8
from sklearn.datasets import make_classification

class SVMClassifier:

    def __init__(self):
        pass


    def fit(self, X, y):
        '''
        Loss function: (1) max soft margin; (2) hinge loss with l2 norm;
        J = 1/2 * ||w||^2 + C \sum_{i=1}^{N} Epsilon_i
        s.t. Epsilon_i >= 0, y_i (W^T * X_i + w_0) >= 1 - Epsilon_i

        (1) Lagrange function, L(w, b, e_i, a_i)
        (2) Dual Problem,  max_{a_i} min_{w, b, e_i} L(w, b, e_i, a_i)
        Solve dual problem, get optimal a_i
        (3) Solve KKT conditions, get optimal w, b, e_i.

        Need SMO.

        '''
        pass


    def predict(self, X):
        pass


def main():
    X, y = make_classification(n_features=4, random_state=0)

    clf = SVMClassifier()
    clf.fit(X, y)
    print(clf.predict(X))


if __name__ == '__main':
    main()