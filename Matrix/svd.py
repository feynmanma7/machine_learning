#encoding:utf-8
import numpy as np
import tensorflow as tf 
np.random.seed(1)


def solve(A, y):
	# Solve Ax = y
	# Return x 

	# A'Ax = A'y
	# cholesky decompostion: LL' = A'A
	# LL'x = A'y, L(sol1) = A'y
	# L'x = sol1

	L = tf.cholesky(tf.matmul(
		tf.transpose(A), A))
	sol1 = tf.matrix_solve(L, 
		tf.matmul(tf.transpose(A), y))
	x = tf.matrix_solve(tf.transpose(L), sol1)

	return x


def svd():
	# R = P * Q
	# R: [m, n]
	# P: [m, k]
	# Q: [k ,n]

	m = 5
	n = 3
	k = 2

	R_val = np.matrix(np.random.random((m, n)))
	P_val = np.matrix(np.random.random((m, k)))
	Q_val = np.matrix(np.random.random((k, n)))

	print('R \n%s' % R_val)

	# tensors
	R = tf.constant(np.matrix(R_val))
	P = tf.Variable(P_val)
	Q = tf.Variable(Q_val)

	# ALS
	n_epoch = 5
	for _ in range(n_epoch):
		# Fixed P, compute Q
		# PQ = R
		Q = solve(P, R)

		# Fixed Q, compute P
		# PQ = R
		# Q'P' = R'
		P = tf.transpose(
			solve(Q, tf.transpose(R)))

	with tf.Session as sess:
		P_eval, Q_eval = sess.run([P, Q])
		print(P_eval, Q_eval)

		R_approx = tf.matmul(P_eval, Q_eval)
		sess.run(R_approx)
		print(R_approx)


def main():
	svd()

if __name__ == '__main__':
	main()