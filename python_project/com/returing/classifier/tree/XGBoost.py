from com.returing.classifier.tree.RegressionTree import generate_data, RegressionTree
from com.returing.metrics.base import compute_accuracy, compute_auc
import numpy as np
np.random.seed(20170430)


class Node:

    row_idx_list = None
    split_attr_idx = None
    split_attr_value = None
    left_child = None
    right_child = None
    is_leaf = None
    weight = None
    depth = None

    def __init__(self):
        self.row_idx_list = []
        self.is_leaf = False
        self.weight = 0.

    def set_row_idx_list(self, row_idx_list):
        self.row_idx_list = row_idx_list

    def add_row_idx(self, row_idx):
        self.row_idx_list.append(row_idx)


class Tree:

    root = None
    max_depth = 5
    gamma = None # regularization parameter for n_leaf
    lambda_ = None # regularization paramter for leaf weight
    min_num_of_leaf = None # min number of samples of leaf node

    def __init__(self,
                 max_depth = 5,
                 lambda_ = 1e-2,
                 gamma = 1e-2,
                 min_num_of_leaf = 1):
        self.max_depth = max_depth
        self.min_num_of_leaf = min_num_of_leaf
        self.lambda_ = lambda_
        self.gamma = gamma


    def _split(self,
               node,
               X,
               split_attr_idx,
               split_attr_value):

        if not node:
            return
        if split_attr_idx < 0:
            return

        node.split_attr_idx = split_attr_idx
        node.split_attr_value = split_attr_value

        node.left_child = Node()
        #node.left_child.weight = node.weight
        node.left_child.depth = node.depth + 1

        node.right_child = Node()
        #node.right_child.weight = node.weight
        node.right_child.depth = node.depth + 1

        for row_idx in node.row_idx_list:
            value = X[row_idx][split_attr_idx]

            if value <= split_attr_value:
                node.left_child.add_row_idx(row_idx)
            else:
                node.right_child.add_row_idx(row_idx)

        if len(node.left_child.row_idx_list) <= 0:
            node.left_child = None
        if len(node.right_child.row_idx_list) <= 0:
            node.right_child = None

        if node.left_child == None and node.right_child == None:
            node.is_leaf = True


    def _choose_best_split_way(self, x, g, lambda_):
        """
        Choose best split way for a specific attribute
        """

        assert x.shape[0] == g.shape[0]

        max_loss_reduction = - np.Infinity
        best_split_value = np.Infinity

        n_samples = x.shape[0]
        for i in range(n_samples - 1):
            # Split between i and i + 1
            left = g[:i+1]
            right = g[i+1:]

            #left_loss = sum(pow(left, 2)) / (left.shape[0] + lambda_)
            #right_loss = sum(pow(right, 2)) / (right.shape[0] + lambda_)

            left_loss = np.asscalar(pow( sum(left), 2 )) / (i+1 + lambda_)
            right_loss = np.asscalar(pow( sum(right), 2)) / (n_samples - i - 1 + lambda_)

            loss = left_loss + right_loss

            if loss > max_loss_reduction:
                max_loss_reduction = loss
                best_split_value = (x[i] + x[i+1]) / 2

        return max_loss_reduction, best_split_value


    def _choose_best_attr_split_way(self,
                                    node,
                                    X,
                                    Y,
                                    Y_pred_last,
                                    attr_idx_list):
        """
        Choose the best split attribute and the corresponding split value.

        Squared Loss: L = 1 / 2 * (y_pred - y) ^ 2

        g = y_pred - y
        H = 1
        """

        max_loss_reduction = - np.Infinity
        best_split_attr_idx = -1
        best_split_value = np.Infinity

        # sub_X = X[node.row_idx_list]
        # sub_Y = Y[node.row_idx_list]
        # sub_Y_pred_last = Y_pred_last[node.row_idx_list]

        for attr_idx in attr_idx_list:
            sorted_idx = np.argsort(X[node.row_idx_list][:, attr_idx])

            sorted_X = X[node.row_idx_list][sorted_idx]
            sorted_Y = Y[node.row_idx_list][sorted_idx]
            sorted_Y_pred_last = Y_pred_last[node.row_idx_list][sorted_idx]

            g = sorted_Y_pred_last - sorted_Y

            # loss_before_split = sum(pow(g, 2)) / (len(node.row_idx_list) + self.lambda_)
            loss_before_split = np.asscalar(pow(sum(g), 2)) / (len(node.row_idx_list) + self.lambda_)

            loss_reduction, split_value = self._choose_best_split_way(
                sorted_X[:, attr_idx], g, self.lambda_)

            loss_reduction = 0.5 * (loss_reduction - loss_before_split) - self.gamma

            if loss_reduction > max_loss_reduction:
                max_loss_reduction = loss_reduction
                best_split_attr_idx = attr_idx
                best_split_value = split_value

        return max_loss_reduction, best_split_attr_idx, best_split_value


    def _build_tree(self, node, X, Y, Y_pred_last, attr_idx_list):
        # === Stop criterion
        if node == None:
            return

        node.is_leaf = (node.depth >= self.max_depth) or \
                       (len(node.row_idx_list) <= 0) or \
                       (len(node.row_idx_list) <= self.min_num_of_leaf) or \
                       (attr_idx_list == None or len(attr_idx_list) <= 0)

        # !!! Set node weight
        g = Y_pred_last[node.row_idx_list] - Y[node.row_idx_list]
        node.weight = - np.asscalar(sum(g)) / (len(node.row_idx_list) + self.lambda_)

        if node.is_leaf == True:
            return


        # === Split
        # Split into left and right child
        # Check split or not
        # Choose best_split_attr_idx and best_split_attr_value

        split_loss_reduction, split_attr_idx, split_attr_value = \
            self._choose_best_attr_split_way(node, X, Y, Y_pred_last, attr_idx_list)

        if split_loss_reduction <= 0.:
            node.is_leaf = True
            return


        # Recursively split left and right child

        self._split(node, X, split_attr_idx, split_attr_value)

        sub_attr_idx_list = attr_idx_list.copy()
        sub_attr_idx_list.remove(split_attr_idx)

        self._build_tree(node.left_child, X, Y, Y_pred_last, sub_attr_idx_list)
        self._build_tree(node.right_child, X, Y, Y_pred_last, sub_attr_idx_list)


    def fit(self, X, Y, Y_pred_last):

        self.root = Node()
        self.root.depth = 1

        row_idx_list = list(range(X.shape[0]))
        self.root.set_row_idx_list(row_idx_list)
        attr_idx_list = list(range(X.shape[1]))

        self._build_tree(self.root, X, Y, Y_pred_last, attr_idx_list)


    def _predict(self, x):
        node = self.root

        while node and not node.is_leaf:
            split_attr_idx = node.split_attr_idx
            split_attr_value = node.split_attr_value

            if split_attr_idx == None or split_attr_idx < 0:
                break

            value = x[split_attr_idx]

            if value <= split_attr_value:
                node = node.left_child
            else:
                node = node.right_child

        return node.weight


    def predict(self, X):
        Y_pred = []

        for x in X:
            y_pred = self._predict(x)
            Y_pred.append([y_pred])

        return np.array(Y_pred).astype(np.float).reshape((X.shape[0], 1))


    def _print_node(self, node):
        print("Node depth=%s, weight=%.4f, is_leaf=%s, "
              "split_attr_idx=%s, split_attr_value=%s" %
              (node.depth,
               node.weight,
               node.is_leaf,
               node.split_attr_idx,
               node.split_attr_value))


    def print_tree(self, node, is_print_leaf=False):
        if node == None:
            return

        if node.is_leaf == is_print_leaf:
            self._print_node(node)

        self.print_tree(node.left_child, is_print_leaf)
        self.print_tree(node.right_child, is_print_leaf)


class XGBoost:

    max_n_trees = 5
    max_depth = 5
    trees = None
    gamma = 1e-2 # regularization parameter for n_leaf
    lambda_ = 1e-2 # leaf weight
    lr = 1e-1 # learning rate

    def __init__(self,
                 max_n_trees=5,
                 max_depth=5,
                 gamma=1e-2,
                 lambda_=1e-2,
                 lr=1e-1):
        self.max_n_trees = max_n_trees
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lr = lr

        self.trees = []


    def fit(self, X, Y):
        n_attrs = X.shape[1]

        self.max_depth = min(self.max_depth, n_attrs)

        """
        rt = RegressionTree()
        rt.fit(X, Y)
        self.trees.append(rt)

        Y_pred_last = rt.predict(X)
        """

        # Y_cur = Y.copy()

        # Bad Initialization for all-0.5 !!!
        # Initialization as 0.5 for each class
        Y_pred_last = np.ones(Y.shape) * 0.5

        # Y_pred_last = np.random.random(Y.shape)

        # Stop Criterion
        for t in range(self.max_n_trees):
            # TODO: Early-stopping

            tree = Tree(max_depth=self.max_depth,
                        gamma=self.gamma,
                        lambda_=self.lambda_)

            tree.fit(X, Y, Y_pred_last)

            self.trees.append(tree)

            Y_pred_t = tree.predict(X)

            Y_pred_last += self.lr * Y_pred_t

            """
            if t == 0:
                Y_pred_last = Y_pred_t * self.lr
            else:
                Y_pred_last += Y_pred_t * self.lr
            """



    def _predict(self, x):

        y_pred = 0.

        for tree in self.trees:
            y_pred_t = tree._predict(x)
            y_pred += y_pred_t

        return y_pred


    def predict(self, X):

        Y_pred = np.zeros((X.shape[0], 1))

        for i in range(len(self.trees)):
            tree = self.trees[i]

            # Y_pred += tree.predict(X) * self.lr
            Y_pred += tree.predict(X) # !!!!

            """
            if i == 0:
                Y_pred += tree.predict(X)
            else:
                Y_pred += tree.predict(X) * self.lr
            """

        return Y_pred


def main():
    X, Y = generate_data()

    clf = XGBoost(max_n_trees=3, gamma=1e-3, lambda_=1e-3, lr=1e-1)
    clf.fit(X, Y)

    Y_pred = clf.predict(X)

    for y, y_pred in zip(Y, Y_pred):
        print(y, y_pred)

    accuracy = compute_accuracy(Y, Y_pred)
    auc = compute_auc(Y, Y_pred)
    print("XGBoost Train accuracy = %.4f " % accuracy)
    print("XGBoost Train auc = %.4f " % auc)


    for i in range(0, len(clf.trees)):
        print("-------Tree %s" % i)
        tree = clf.trees[i]
        tree.print_tree(tree.root, is_print_leaf=True)
        #tree.print_tree(tree.root, True)
        print("\n\n")




if __name__ == '__main__':
    main()