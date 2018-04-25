#encoding:utf-8
import numpy as np
import sys

class CARTRegressor:

    '''
    node:
    # datasets
    # is_leaf
    # response: predict_value = avg(datasets)
    # split_feature
    # split_value
    # left_node
    # right_node
    '''

    def __init__(self,
                 max_depth=4,
                 min_leaf_samples = 1):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.tree = {}


    def _l2loss(self, y):
        if len(y) == 0:
            return 0

        mean = sum(y) / len(y)
        loss = 0

        for i in range(len(y)):
            loss += (y[i] - mean) ** 2

        return loss / 2


    def _compute_loss(self, x, y, split_point):
        left_y = []
        right_y = []

        for i in range(len(x)):
            if x[i] <= split_point:
                left_y.append(y[i])
            else:
                right_y.append(y[i])

        # least square loss
        left_loss = self._l2loss(left_y)
        right_loss = self._l2loss(right_y)

        return left_loss + right_loss


    def _get_split_point(self, X, y, feature):

        # Get unique sorted list.
        sorted_list = np.array(list(set(sorted(X[:, feature]))))

        best_split_point = 0
        min_loss = sys.float_info.max

        for i in range(sorted_list.size - 1):
            # Use mean of the sorted as split point,
            # compute loss,
            # select split point with minimum loss.

            split_point = (sorted_list[i] + sorted_list[i+1]) / 2
            loss = self._compute_loss(X[:, feature], y, split_point)

            if loss < min_loss:
                min_loss = loss
                best_split_point = split_point

        return best_split_point, min_loss


    def _compute_split(self, datasets):
        X, y = datasets

        best_feature = 0
        best_split_point = 0

        if len(X) == 0:
            return -1, -1

        num_feature = X.shape[1]

        min_loss = sys.float_info.max

        for feature in range(num_feature):
            split_point, loss = self._get_split_point(X, y, feature)

            if loss < min_loss:
                min_loss = loss
                best_feature = feature
                best_split_point = split_point

        return best_feature, best_split_point


    def _split(self, datasets, feature, split_point):

        X, y = datasets

        feature = int(feature)
        split_point = float(split_point)

        left_X = []
        left_y = []
        right_X = []
        right_y = []

        for i in range(len(X)):
            data = X[i]
            if float(data[feature]) <= split_point:
                left_X.append(data)
                left_y.append(y[i])
            else:
                right_X.append(data)
                right_y.append(y[i])

        left_datasets = np.array(left_X), np.array(left_y)
        right_datasets = np.array(right_X), np.array(right_y)

        return left_datasets, right_datasets


    def _build_tree(self, datasets, depth):

        tree = {}
        tree['is_leaf'] = False

        X, y = datasets

        if depth == self.max_depth or len(datasets) < self.min_leaf_samples or len(datasets) == 0:
            tree['is_leaf'] = True
            if len(y) == 0:
                tree['response'] = 0
            else:
                tree['response'] = sum(y) / len(y)

            return tree

        feature, split_point = self._compute_split(datasets)

        tree['feature'] = feature
        tree['split_point'] = split_point

        left_datasets, right_datasets = self._split(datasets, feature, split_point)

        tree['left_node'] = self._build_tree(left_datasets, depth + 1)
        tree['right_node'] = self._build_tree(right_datasets, depth + 1)

        return tree


    def fit(self, X, y):
        '''
        (1) select best feature
        (2) select best split of the feature

        Regression problem, using least square error.

        '''

        datasets = [X, y]

        self.tree = self._build_tree(datasets, depth=1)


    def _print_tree(self, tree, depth):

        if tree['is_leaf'] == True:
            print('Leaf Node, depth=%s, response=%s' % (depth, tree['response']))

        else:
            print('feature=%s, split_point=%s' % (tree['feature'], tree['split_point']))

            print('left_tree')
            self._print_tree(tree['left_node'], depth + 1)

            print('right_tree')
            self._print_tree(tree['right_node'], depth + 1)


    def print_tree(self):
        self._print_tree(self.tree, depth=1)


    def _predict(self, tree, x):
        if tree['is_leaf'] == True:
            return tree['response']

        if x[tree['feature']] <= tree['split_point']:
            return self._predict(tree['left_node'], x)
        else:
            return self._predict(tree['right_node'], x)


    def predict(self, X):
        tree = self.tree
        preds = []

        for x in X:
            pred = self._predict(tree, x)
            preds.append(pred)

        return np.array(preds)


class CARTClassifier:

    def __init__(self):
        pass


    def fit(self):
        pass


    def predict(self):
        pass



def main():
    input_path = '../data/iris/iris_binary_class.txt'

    data = np.loadtxt(input_path)
    X = data[:, :4]
    y = data[:, 4]

    X = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
    y = [1, 2, 3, 4, 5]

    X = np.array(X)
    y = np.array(y)

    reg = CARTRegressor()
    reg.fit(X, y)
    #reg.print_tree()
    print(reg.predict(X))


pass

if __name__ == '__main__':
    main()
