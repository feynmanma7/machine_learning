from ID3 import ID3
from my_util import generate_data
import numpy as np
np.random.seed(20170430)

class C45(ID3):

    def __init__(self):
        super(C45, self).__init__()


    def _compute_gain_ratio(self, X, Y, attr_idx, row_idx_list):
        """
        :param X:
        :param Y:
        :param attr_idx:
        :param row_idx_list:
        :return: gain_ratio

        gain_ratio(X, attr) = info_gain(X, attr) / intrinsic_value(attr)

        key_set = unique(X[attr])

        intrinsic_value(attr) = - sum_{key}^{key_set} ratio(key) \log ratio(key)

        ratio(key) = num_key / num_sample

        """

        entropy = self._compute_entropy(Y, row_idx_list)

        info_gain = entropy

        key_set = np.unique(X[row_idx_list][:, attr_idx])

        intrinsic_value = 0.

        for key in key_set:
            key_row_idx_list = list(np.where(X[row_idx_list][:, attr_idx] == key))
            key_ratio = len(key_row_idx_list) / len(row_idx_list)
            key_entropy = self._compute_entropy(Y, key_row_idx_list)

            info_gain -= key_ratio * key_entropy

            intrinsic_value += - key_ratio * np.log(key_ratio)

        gain_ratio = info_gain / intrinsic_value

        return gain_ratio

    def _choose_best_split_attr_idx(self, X, Y, attr_idx_list, row_idx_list):

        if attr_idx_list == None or len(attr_idx_list) <= 0:
            return -1

        max_gain_ratio = -1
        best_attr_idx = -1

        for attr_idx in attr_idx_list:
            gain_ratio = self._compute_gain_ratio(X, Y, attr_idx, row_idx_list)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_attr_idx = attr_idx

        return best_attr_idx



def main():

    X, Y = generate_data()

    c45 = C45()
    c45.fit(X, Y)
    results = c45.predict(X)
    print(results)


if __name__ == '__main__':
    main()