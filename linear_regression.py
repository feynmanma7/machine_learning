# encoding:utf-8
from functools import reduce
import numpy as np
import sys

np.random.seed(20170430)
from sklearn import datasets


class LinearRegression():
    def __init__(self,
                 alpha=1e-3,
                 reg=1e-4,
                 n_epoch=10000,
                 threshold=1e-4):
        self.alpha = alpha
        self.reg = reg
        self.n_epoch = n_epoch
        self.threshold = threshold


    def _compute_loss(self, X, y):
        loss = 0

        for i in range(len(X)):
            pred = self.predict(X[i])
            error = pred - y[i]
            loss += error ** 2

        loss += 0.5 * self.reg * np.sum(
            np.dot([self.coef_, self.intercept_],
                   [self.coef_, self.intercept_]))

        return 0.5 / len(X) * loss


    def fit(self, X, y):

        '''
        f = W * X + b
        loss = 0.5 * sum_{i=1}^{N} (f(i) - y[i]) ** 2 + 0.5 * reg * w ** 2

        SGD
        delta_{w_j} = (f(i) - y[i]) * X[i][j] + reg * w_j
        delta_{b} = (f[i] - y[i]) + reg * b

        w_j = w_j - alpha * delta_{w_j}
        b = b - alpha * delta_{b}
        '''

        self.coef_ = np.random.random(len(X[0]))
        self.intercept_ = np.random.random()

        pre_loss = 0

        for epoch in range(self.n_epoch):

            for i in range(len(X)):
                pred = self.predict(X[i])
                error = pred - y[i]

                self.intercept_ -= self.alpha * (
                    error + self.reg * self.intercept_)

                for j in range(self.coef_.size):
                    self.coef_[j] -= self.alpha * (
                        error * X[i][j] + self.reg * self.coef_[j])

            # compute loss
            cur_loss = self._compute_loss(X, y)
            delta_loss = abs(pre_loss - cur_loss)
            pre_loss = cur_loss

            print('epoch%s, loss: %s, coefs: %s, intercepts: %s'
                  % (epoch, cur_loss, self.coef_, self.intercept_))


            # check covergence
            if delta_loss < self.threshold:
                return



    def predict(self, x):
        return np.dot(self.coef_, x) + self.intercept_



def main():
    reg = LinearRegression()

    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    reg.fit(diabetes_X_train, diabetes_y_train)

    #reg.fit(X, y)
    print(reg.coef_, reg.intercept_)


if __name__ == '__main__':
    main()