import queue
import numpy as np
np.random.seed(20170430)

class Node:
    split_attr_idx = None
    split_attr_value = None
    left_child = None
    right_child = None

    # row index list of samples in current None, memory costly
    row_idx_list = None
    n_samples = None
    # impurity = None # n_positive / n_samples
    is_leaf = None
    score = None  # sum(y) / n_y
    depth = None

    def __init__(self):
        self.split_attr_idx = -1
        self.is_leaf = False
        self.row_idx_list = []
        self.score = 0.
        self.depth = 0

    def set_row_idx_list(self, row_idx_list):
        self.row_idx_list = row_idx_list

    def add_row_idx(self, row_idx):
        self.row_idx_list.append(row_idx)


class RegressionTree():

    max_depth = 5
    min_num_of_leaf = 1
    root = None
    impurity_threshold = 0.8

    def __init__(self):
        pass

    def _compute_region_loss(self, Y_region):
        # Variance
        Y_pred = np.average(Y_region)
        return np.sum(np.subtract(Y_region, Y_pred))

    def _find_best_attr_split_value(self, node, X, Y, attr_idx):
        row_idx_list = node.row_idx_list

        samples = X[row_idx_list][:, attr_idx]

        sorted_idx = np.argsort(samples)
        sorted_X = samples[sorted_idx]
        sorted_Y = Y[row_idx_list][sorted_idx]

        best_split_idx = -1 # split between best_split_idx and best_split_idx + 1
        min_total_loss = np.Infinity

        for i in range(sorted_Y.shape[0] - 1):
            left_loss = self._compute_region_loss(Y[:i+1])
            right_loss = self._compute_region_loss(Y[i+1:])
            total_loss = left_loss + right_loss

            if total_loss < min_total_loss:
                min_total_loss = total_loss
                best_split_idx = i

        best_split_value = np.average((sorted_X[best_split_idx], \
                                       sorted_X[best_split_idx+1]))

        return best_split_value, min_total_loss


    def _find_best_split_way(self, node, X, Y, attr_idx_list):

        min_total_loss = np.Infinity
        best_split_attr_idx = -1
        best_split_attr_value = np.Infinity

        for attr_idx in attr_idx_list:
            attr_split_value, attr_split_loss = self._find_best_attr_split_value(
                node, X, Y, attr_idx)

            if attr_split_loss < min_total_loss:
                min_total_loss = attr_split_loss
                best_split_attr_idx = attr_idx
                best_split_attr_value = attr_split_value

        return best_split_attr_idx, best_split_attr_value


    def _split(self, node, X, split_attr_idx, split_attr_value):

        if node == None or node.row_idx_list == None \
                or len(node.row_idx_list) <= 0:
            return


        for row_idx in node.row_idx_list:
            x = X[row_idx]
            value = x[split_attr_idx]

            if value <= split_attr_value:

                if node.left_child == None:
                    node.left_child = Node()
                    node.left_child.depth = node.depth + 1

                    # Pass parent node's score to child
                    node.left_child.score = node.score

                node.left_child.add_row_idx(row_idx)
            else:

                if node.right_child == None:
                    node.right_child = Node()
                    node.right_child.depth = node.depth + 1

                    # Pass parent node's score to child
                    node.right_child.score = node.score

                node.right_child.add_row_idx(row_idx)


    def _build_tree(self, node, X, Y, attr_idx_list):

        """

        :param node:
        :param X:
        :param Y:
        :param row_idx_list:
        :param attr_idx_list:
        :return:

        Stop criterion:
        1. Is_leaf marked in other places.
        2. Meet min_num_of_leaf or max_depth

        For regression tree, no impurity
        # or impurity_threshold

        """

        if node == None or node.is_leaf == True or node.row_idx_list == None \
                or attr_idx_list == None or len(attr_idx_list) <= 0:
            node.is_leaf = True
            return

        # node.impurity = n_positive * 1.0 / n_samples

        node.is_leaf = (len(node.row_idx_list) <= self.min_num_of_leaf) or \
                       (node.depth >= self.max_depth)
                      #  or (node.impurity >= self.impurity_threshold)

        # Compute score

        node.n_samples = len(node.row_idx_list)
        node.score = np.sum(Y[node.row_idx_list]) / node.n_samples

        if node.is_leaf == True:
            return

        #=== Split Node Recursively.

        node.split_attr_idx, node.split_attr_value = \
            self._find_best_split_way(node, X, Y, attr_idx_list)

        # split
        self._split(node, X, node.split_attr_idx, node.split_attr_value)

        # build tree of children
        child_attr_idx_list = attr_idx_list.copy()
        child_attr_idx_list.remove(node.split_attr_idx)
        self._build_tree(node.left_child, X, Y, child_attr_idx_list)
        self._build_tree(node.right_child, X, Y, child_attr_idx_list)



    def fit(self, X, Y):

        self.root = Node()
        row_idx_list = list(range(X.shape[0]))
        attr_idx_list = list(range(X.shape[1]))

        self.root.set_row_idx_list(row_idx_list)
        self.root.depth = 1

        self._build_tree(self.root, X, Y, attr_idx_list)


    def _predict(self, x):

        node = self.root

        while node and not node.is_leaf:

            split_attr_idx = node.split_attr_idx
            split_attr_value = node.split_attr_value

            if x[split_attr_idx] <= split_attr_value:
                node = node.left_child
            else:
                node = node.right_child

        return node.score


    def predict(self, X):

        results = []
        for x in X:
            results.append(self._predict(x))

        return results

    def print_tree(self):

        node = self.root

        # bread-first transverse
        q = queue.Queue()

        q.put(node)

        while not q.empty():
            node = q.get()
            if node == None:
                continue
            if node.is_leaf:
                print("depth=%s, score=%s" % (node.depth, node.score))
            else:
                print("depth=%s, split_idx=%s, split_value=%s" % \
                    (node.depth, node.split_attr_idx, node.split_attr_value))

            if node.left_child:
                q.put(node.left_child)
            if node.right_child:
                q.put(node.right_child)

def generate_data():
    input_path = '/Users/flyingman/Data/bi_iris'

    X = []
    Y = []

    for line in open(input_path, 'r'):
        buf = line[:-1].split(",")
        x = buf[:-1]
        y = buf[-1]
        X.append(x)
        Y.append([y])

    return np.array(X).astype(np.float), np.array(Y).astype(np.float)

def main():
    X, Y = generate_data()

    print(X.shape)
    print(Y.shape)

    reg_tree = RegressionTree()
    reg_tree.fit(X, Y)
    results = reg_tree.predict(X)

    for result, y in zip(results, Y):
        print(result, y)
    #print(results)

    # reg_tree.print_tree()

if __name__ == '__main__':
    main()
