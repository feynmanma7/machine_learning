#encoding:utf-8
import numpy as np
from DecisionTree import CARTRegressor

class GBDTRegressor:

    def __init__(self,
                 n_estimators=100,
                 max_depth=4,
                 min_leaf_samples=1,
                 learning_rate=0.1):
        # num of trees
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.learning_rate = learning_rate


    def fit(self, X, y):
        '''
        For regression, using squared loss.
        f_m(x) = f_{m-1}(x) + \beta * h(x)

        According to Taylor Expansion,
        loss(y, y^m) = loss(y, y^{m-1} + \beta * h(x))
             = loss(y, y^{m-1}) + \partial{loss(y, y^{m-1})} * \beta * h(x)

        To Get a Minimization of loss(y, y^m),
        could let \beta * h(x) be - \partial{loss(y, y^{m-1})}.
        h(x) = argmin_{\beta, h} (\beta * h(x) - \partial{loss(y, y^{m-1})}) ** 2

        If h(x) is a cart-based tree, let the tree to fit \partial{loss(y, y^{m-1})}.
        which is the Residual in the case of squared loss

        '''

        self.trees = []

        for num in range(self.n_estimators):
            reg = CARTRegressor(self.max_depth, self.min_leaf_samples)
            reg.fit(X, y)
            self.trees.append(reg)

            y_pred = reg.predict(X)

            # residual
            y = y - self.learning_rate * y_pred



    def predict(self, X):

        preds = np.zeros(X.shape[0])

        for reg in self.trees:
            preds += self.learning_rate * reg.predict(X)

        return preds



def main():

    X = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    y = [1, 2, 3, 4, 5]

    X = np.array(X)
    y = np.array(y)

    reg = GBDTRegressor()
    reg.fit(X, y)
    print(reg.predict(X))
    print(len(reg.trees))



if __name__  == '__main__':
    main()