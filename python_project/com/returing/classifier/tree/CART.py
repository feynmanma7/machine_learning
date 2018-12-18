from ID3 import ID3
from my_util import generate_data
import numpy as np
np.random.seed(20170430)


class CART(ID3):

    def __init__(self):
        super(CART, self).__init__()

    def _compute_gini_index(self, X, Y, row_idx_list):

        """

        :param X:
        :param Y:
        :param attr_idx:
        :param row_idx_list:
        :return:

        gini_index = \sum_{n_1 = 1}^{num_Y} \sum_{n_2 != n_1}^{num_Y} (p(n_1) * p(n_2))
        = 1 - \sum_{n = 1}^{num_Y} p_n ^ 2
        """

        gini_index = 1.

        label_set = np.unique(Y[row_idx_list])

        n_sample = len(row_idx_list)

        for label in label_set:
            p_label = Y[row_idx_list][Y[row_idx_list] == label].shape[0] * 1. / n_sample
            gini_index -= p_label ** 2

        return gini_index


    def _compute_attr_gini_index(self, X, Y, attr_idx, row_idx_list):

        """

        :param X:
        :param Y:
        :param attr_idx:
        :param row_idx_list:
        :return:

        gini_index = \sum_{n_1 = 1}^{num_Y} \sum_{n_2 != n_1}^{num_Y} (p(n_1) * p(n_2))
        = 1 - \sum_{n = 1}^{num_Y} p_n ^ 2
        """

        gini_index = 0.

        key_set = np.unique(X[row_idx_list][:, attr_idx])

        for key in key_set:

            key_row_idx_list = list(Y[row_idx_list][Y[row_idx_list] == key])

            key_gini_index = self._compute_gini_index(X, Y, key_row_idx_list)

            key_ratio = len(key_row_idx_list) / \
                len(row_idx_list)

            gini_index += key_ratio * key_gini_index

        return gini_index


    def _choose_best_split_attr_idx(self, X, Y, attr_idx_list, row_idx_list):

        if attr_idx_list == None or len(attr_idx_list) <= 0:
            return -1

        max_gini_index = -1
        best_attr_idx = -1

        for attr_idx in attr_idx_list:
            gini_index = self._compute_attr_gini_index(
                X, Y, attr_idx, row_idx_list)
            if gini_index > max_gini_index:
                max_gini_index = gini_index
                best_attr_idx = attr_idx

        return best_attr_idx



def main():
    X, Y = generate_data()

    cart = CART()
    cart.fit(X, Y)
    results = cart.predict(X)
    print(results)


if __name__ == '__main__':
    main()