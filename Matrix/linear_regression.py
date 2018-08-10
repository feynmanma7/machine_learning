#encoding:utf-8

import numpy as np 
import tensorflow as tf 
np.random.seed(7)

def linear_regression():
	# To solve Ax + b = y
	A_val = np.linspace(0, 10, 100)
	b_val = np.repeat(1, 100)
	y_val = A_val + np.random.normal(0, 1, 100)

	A_col = np.transpose(np.matrix(A_val))
	b_col = np.transpose(np.matrix(b_val))
	A = np.column_stack((A_col, b_val))
	y = np.transpose(np.matrix(y_val))

	A_tensor = tf.constant(A)
	y_tensor = tf.constant(y)

	# To solve Ax = y
	# A'Ax = A'y
	# Use Cholesky Decomposition, A'A = LL'
	# LL'x = A'y
	# Solve L(sol1) = A'y
	# Solve L'x = sol1
	L = tf.cholesky(tf.matmul(
		tf.transpose(A_tensor), A_tensor))
	sol1 = tf.matrix_solve(L, tf.matmul(
		tf.transpose(A_tensor), y_tensor))
	sol2 = tf.matrix_solve(tf.transpose(L), sol1)

	with tf.Session() as sess:
		sol = sess.run(sol2)
		print(sol)


def main():
	linear_regression()

if __name__ == '__main__':
	main()
