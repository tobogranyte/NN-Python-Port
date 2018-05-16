def predict(Theta1, Theta2, X):

	import numpy as np
	from sigmoid import sigmoid

	m = X.shape[0]
	num_labels = Theta2.shape[0]

	p = np.zeros((m, 1))

	h1 = sigmoid(np.dot(np.append(np.ones((m, 1)), X, axis=1), Theta1.T))
	h2 = sigmoid(np.dot(np.append(np.ones((m, 1)), h1, axis=1), Theta2.T))

	p = np.mod(np.argmax(h2, axis=1) + 1, 10)

	return p