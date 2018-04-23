#encoding:utf-8
import numpy as np
import math

def sigmoid(x):
	return 1. / (1 + math.exp(-x))

def precision_recall(true_labels, pred_labels):
    n_pred_right = np.sum(list(map(
        lambda x, y: 1 if x == y and x == 1 else 0, true_labels, pred_labels)))
    precision = n_pred_right / len(true_labels)
    recall = n_pred_right / np.sum(true_labels[np.where(true_labels == 1)])

    print('precison= %s, recall= %s' % (precision, recall))


class LogisticeRegression():

    def __init__(self,
		alpha=1e-4,
		reg=1e-3,
		n_epochs=5000,
		threshold=1e-4
		):
        self.alpha = alpha
        self.reg = reg
        self.n_epochs = n_epochs
        self.threshold = threshold


    def _compute_loss(self, X, y):
        loss = 0

        for i in range(X.shape[0]):
            pred = self.predict_prob(X[i])[0]
            if 1 - pred < 1e-8:
                loss += y[i] * math.log(pred, 2) + (1 - y[i]) * math.log(1e-8, 2)
            else:
                loss += y[i] * math.log(pred, 2) + (1 - y[i]) * math.log(1 - pred, 2)
        return loss / X.shape[0]


    def fit(self, X, y):
        '''
		Binomial distribuiton, let loss be negative log likelihood.
		loss = \log(\sum_{i=1}^{N} \theta^{y_i} (1-\theta)^(1-y_i))
		= \sum_{i=1}^{N} y_i * \log \theta + (1-y_i) \log(1-\theta)

		\theta is sigmoid function:
		\theta = \frac{1}{1 + exp(-W^T * X)}

		Stochasic Gradient of w_j (j=0, 1, ..., M-1) and b.
		\sigmoid' = \sigmoid(1-\sigmoid)
		delta_{w_j} = (y_i - \sigmoid(X_i)) X_{ij}
		delta_{b} = (y_i - \sigmoid(X_i))
        '''

        # Initializiation of parameters.
        self.coef_ = np.random.random(X.shape[1])
        self.intercept_ = np.random.random()

        pre_loss = 0

        for epoch in range(self.n_epochs):
			
            for i in range(len(X)):

                pred = self.predict_prob(X[i])[0]

                for j in range(X.shape[1]):
                    self.coef_[j] -= self.alpha * (
					 (y[i] - pred) * X[i][j])
                self.intercept_ -= self.alpha * (
					y[i] - pred)

            # compute loss
            cur_loss = self._compute_loss(X, y)
            delta_loss = abs(pre_loss - cur_loss)

            print('epoch%s, loss: %s, delta_loss: %s coefs: %s, intercepts: %s'
                  % (epoch, cur_loss, delta_loss, self.coef_, self.intercept_))

            pre_loss = cur_loss

            # check convergence
            if delta_loss < self.threshold:
                return
	

    def predict_prob(self, X):
        results = []
        for x in X:
            results.append(sigmoid(np.sum(np.dot(self.coef_, x)) + self.intercept_))
        return results


    def predict(self, X):
        results = []
        for x in X:
            prob = sigmoid(np.sum(np.dot(self.coef_, x)) + self.intercept_)
            label = 1 if prob > 0.5 else 0
            results.append(label)
        return results


def main():
    input_path = '../data/iris/iris_binary_class.txt'

    data = np.loadtxt(input_path)
    X = data[:, :4]
    y = data[:, 4]

    lr = LogisticeRegression()
    lr.fit(X, y)

    pred_labels = lr.predict(X)
    precision_recall(y, pred_labels)


if __name__ == '__main__':
	main()