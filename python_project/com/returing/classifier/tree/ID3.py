#encoding:utf-8
from my_util import generate_data

import numpy as np
np.random.seed(20170430)


class Node:

    row_idx_list = [] # data
    split_attr_idx = 0
    is_leaf_node = False
    child_node = {}   # k-v pointers to child node
    score = None

    def __init__(self):
        self.row_idx_list = []
        self.is_leaf_node = False
        self.split_attr_idx = -1
        self.child_node = None
        self.score = -1.

    def set_row_idx_list(self, row_idx_list):
        self.row_idx_list = row_idx_list

    def add_row_idx(self, row_idx):
        self.row_idx_list.append(row_idx)


class ID3:

    root = None
    max_depth = 3
    min_leaf_num = 1

    split_attr_idx_list = []

    def __init__(self):
        self.root = None

    def _compute_entropy(self, Y, row_idx_list):
        label_set = np.unique(Y[row_idx_list])
        num_sample = len(row_idx_list)

        entropy = 0.
        for label in label_set:
            p = Y[row_idx_list][Y[row_idx_list] == label].shape[0] / num_sample
            entropy += - p * np.log(p)

        return entropy

    def _compute_info_gain(self, X, Y, attr_idx, row_idx_list):
        """
        :param X:
        :param Y:
        :param attr_idx:
        :param row_idx_list:
        :return: gain_ratio

        info_gain = Entropy(X) - \sum_{key}^{key_set} ratio(key) Entropy(X[key])
        key_set = X[:, attr_idx]
        ratio(key) = num_key / num_sample
        """

        entropy = self._compute_entropy(Y, row_idx_list)

        info_gain = entropy

        key_set = np.unique(X[row_idx_list][:, attr_idx])

        for key in key_set:
            key_row_idx_list = list(np.where(X[row_idx_list][:, attr_idx] == key))
            key_ratio = len(key_row_idx_list) / len(row_idx_list)
            key_entropy = self._compute_entropy(Y, key_row_idx_list)

            info_gain -= key_ratio * key_entropy

        return info_gain


    def _choose_best_split_attr_idx(self, X, Y, attr_idx_list, row_idx_list):

        if attr_idx_list == None or len(attr_idx_list) <= 0:
            return -1

        max_info_gain = -1
        best_attr_idx = -1

        for attr_idx in attr_idx_list:
            info_gain = self._compute_info_gain(X, Y, attr_idx, row_idx_list)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr_idx = attr_idx

        return best_attr_idx

    def _build_tree(self, node, attr_idx_list, X, Y):

        if node.is_leaf_node == True:
            return

        """
        If 
         
        1. Current node is empty, 
        2. Number of records is less than the threshold,
        3. All of the records belong to the same class,
        
        Mark current node as leaf, Stop split. 
        """

        if node.row_idx_list == None:
            node.is_leaf_node = True
            return

        # Set score of current node.
        num_sample = len(node.row_idx_list)
        num_positive = sum(Y[node.row_idx_list])
        node.score = num_positive * 1.0 / num_sample

        if len(node.row_idx_list) <= self.min_leaf_num:
            node.is_leaf_node = True
            return

        if num_sample == num_positive:
            node.is_leaf_node = True
            return

        """
        Split 
        """
        split_attr_idx = self._choose_best_split_attr_idx(
            X, Y, attr_idx_list, node.row_idx_list)

        if split_attr_idx < 0:
            return

        node.split_attr_idx = split_attr_idx

        # Split node
        node.child_node = {}

        for row_idx in node.row_idx_list:
            key = X[row_idx][split_attr_idx]

            if key not in node.child_node:
                node.child_node[key] = Node()
                # Pass parent's score to child node.
                node.child_node[key].score = node.score

            node.child_node[key].add_row_idx(row_idx)

        sub_attr_idx_list = attr_idx_list.copy()
        sub_attr_idx_list.remove(split_attr_idx)
        for key in node.child_node:
            self._build_tree(node.child_node[key], sub_attr_idx_list, X, Y)


    def fit(self, X, Y):

        self.root = Node()
        self.root.set_row_idx_list(list(range(X.shape[0])))
        num_sample = Y.shape[0]
        num_positive = sum(Y)

        # Pass parent node's score to child.
        self.root.score = num_positive / num_sample

        attr_idx_list = list(range(X.shape[1]))

        self._build_tree(self.root, attr_idx_list, X, Y)


    def _print_node(self, node):

        if node == None:
            return

        if node.is_leaf_node == False:
            print(node.score)

            if node.child_node == None:
                return

            for key in node.child_node:
                self._print_node(node.child_node[key])


    def print_tree(self):
        node = self.root
        self._print_node(node)

    def _predict(self, x):

        node = self.root

        while node.is_leaf_node == False:

            if node.child_node == None:
                return node.score

            key = x[node.split_attr_idx]
            node = node.child_node[key]

        return node.score


    def predict(self, X):

        results = []
        for x in X:
            results.append(self._predict(x))

        return results



def main():
    X, Y = generate_data()

    id3 = ID3()
    id3.fit(X, Y)
    results = id3.predict(X)
    print(results)


if __name__ == '__main__':
    main()