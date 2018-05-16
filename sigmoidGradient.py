def sigmoidGradient(z):

	import numpy as np
	from sigmoid import sigmoid

	g = np.zeros((z.shape))

	g = sigmoid(z) * (1 - sigmoid(z))

	return g