def computeNumericalGradient (J, theta):
	import numpy as np

	print(theta.shape)
	numgrad = np.zeros((theta.shape))
	perturb = np.zeros((theta.shape))

	e = 1e-4
	for p in range(theta.size):
		perturb[p] = e
		loss1 = J(theta - perturb)[0]
		loss2 = J(theta + perturb)[0]
		numgrad[p] = (loss2 - loss1) / (2 * e)
		perturb[p] = 0

	return numgrad