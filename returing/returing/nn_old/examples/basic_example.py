from returing.data.binary_classification import generate_data
import numpy as np
np.random.seed(20170430)


def main():

    """
    X: n_samples * dim_input
    Y: n_samples * 1
    """

    X, Y = generate_data()
    dim_H = 1
    dim_output = 1
    learning_rate=1e-2
    n_iter = 10

    n_samples, dim_input = X.shape

    # Predict
    # Y_pred = model.predict(X)



if __name__ == '__main__':
    main()