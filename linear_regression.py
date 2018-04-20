#encoding:utf-8
from functools import reduce
import numpy as np
import sys
np.random.seed(20170430)


class LinearRegression():

	def __init__(self, 
		alpha=0.01, 
		max_iter=100, 
		threshold=1e-4, 
		dim=2,
		batch_size=10):
		self.alpha = alpha
		self.max_iter = max_iter
		self.threshold = threshold
		self.coef_ = np.random.random(dim)
		self.intercept_ = np.random.random()
		self.batch_size = batch_size


	def _compute_loss(self, X, y):

		loss = 0

		for i in range(len(X)):
			h = np.dot(X[i], self.coef_) + self.intercept_

			loss += (h - y[i]) ** 2

		return loss 

		'''
		return np.sum(list(map(
			lambda a, b: (np.dot(self.coef_, a) + self.intercept_ - b) ** 2, X, y)))
		'''


	def fit(self, X, y):

		'''
		f = W * X + b
		loss = sum_{i=1}^{N} (f(i) - y[i]) ** 2

		SGD
		delta_{w_j} = 2 (f(i) - y[i]) * X[i][j]
		delta_{b} = 2 (f[i] - y[i])

		w_j = w_j - alpha * delta_{w_j}
		b = b - alpha * delta_{b}

		'''

		pre_loss = 0

		num = 0
		delta_b = 0
		delta_w = np.zeros(self.coef_.size)

		for iter in range(self.max_iter):

			for i in range(len(X)):
				
				h = np.dot(self.coef_, X[i]) + self.intercept_

				if num < self.batch_size:
					num += 1
					delta_b += 2. * (h - y[i])

					for j in range(self.coef_.size):
						delta_w[j] += 2. * (h - y[i]) * X[i][j]

				else:

					for j in range(self.coef_.size):
						self.coef_[j] -= self.alpha * delta_w[j] / self.batch_size
						delta_w[j] = 0
					self.intercept_ -=  self.alpha * delta_b / self.batch_size
					delta_b = 0

					num = 0


					cur_loss = self._compute_loss(X, y)
					delta_loss = cur_loss - pre_loss

					pre_loss = cur_loss

					if delta_loss < self.threshold:
						return


	def predict(self, X):
		pass


def main():
	reg = LinearRegression()
	X = [[0., 0.], [1., 1.], [2., 2.]]
	y = [0, 1, 2]
	reg.fit(X, y)
	print(reg.coef_, reg.intercept_)
	print(reg._compute_loss(X, y))

if __name__ == '__main__':
	main()