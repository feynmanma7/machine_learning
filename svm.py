#encoding:utf-8
from sklearn.datasets import make_classification

class SVMClassifier:

    def __init__(self):
        pass


    def fit(self, X, y):
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