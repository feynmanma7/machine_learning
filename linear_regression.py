#encoding:utf-8
from functools import reduce
import numpy as np
import sys
np.random.seed(20170430)

class LinearRegression():

	alpha = 1e-4
	max_iter = 100
	threshold = 1e-4
	coef_ = []
	intercept_ = 0

	def __init__(self, 
		alpha=1e-4, 
		max_iter=100, 
		threshold=1e-6, 
		dim=2):
		self.alpha = alpha
		self.max_iter = max_iter
		self.threshold = threshold
		self.coef_ = np.random.random(dim)
		self.intercept_ = np.random.random()
		#self.intercept = 0.


	def _compute_loss(self, X, y):
		return np.sum(list(map(
			lambda a, b: (np.sum(self.coef_ * a) + self.intercept_ - b) ** 2, X, y)))


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

		for iter in range(self.max_iter):

			for i in range(len(X)):
				
				f = np.sum(self.coef_ * X[i]) + self.intercept_

				delta_b = 2. * (f - y[i])

				for j in range(self.coef_.size):
					delta_w = delta_b * X[i][j]
					self.coef_[j] = self.coef_[j] - self.alpha * delta_w
				
				self.intercept_ = self.intercept_ - self.alpha * delta_b

			cur_loss = self._compute_loss(X, y)

			pre_loss = cur_loss

			delta_loss = cur_loss - pre_loss

			if delta_loss < self.threshold:
				return


	def predict(self, X):
		pass


def main():
	reg = LinearRegression()
	X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4, 4], [5, 5], [6, 6]]
	y = [0, 1, 2, 3, 4, 5, 6]
	reg.fit(X, y)
	print(reg.coef_, reg.intercept_)
	print(reg._compute_loss(X, y))

if __name__ == '__main__':
	main()