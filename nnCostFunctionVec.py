def nnCostFunctionVec (nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam, returnType=''):

	import numpy as np
	from sigmoid import sigmoid
	from sigmoidGradient import sigmoidGradient

	Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1, order = 'F')
	Theta2 = nn_params[(hidden_layer_size *(input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size +1), order = 'F')

	(m, n) = X.shape

	J = 0
	Theta1_grad = np.zeros((Theta1.shape))
	Theta2_grad = np.zeros((Theta2.shape))
	grad = 0

	ident = np.eye(Theta2.shape[0])

	yNodes = ident[y.flatten()].T
	yNodes = np.append(yNodes[1:,], yNodes[0:1,], axis=0)


	X = np.append(np.ones((m,1)), X, axis=1)

	if returnType == '' or returnType == 'J':
		h = sigmoid(np.dot(Theta2, (np.append(np.ones((1, m)), sigmoid(np.dot(Theta1, X.T)), axis=0))))
		J = np.sum(-yNodes * np.log(h)-((1 - yNodes) * np.log(1 - h)))/m + lam * (np.sum(np.square(Theta2[:,1:])) + np.sum(np.square(Theta1[:,1:])))/(2*m)

	if returnType == '' or returnType == 'grad':
		delta3 = sigmoid(np.dot(Theta2, np.append(np.ones((1,m)), sigmoid(np.dot(Theta1, X.T)), axis=0)))		 - yNodes
		delta2 = (np.dot(Theta2.T, delta3) * sigmoidGradient(np.append(np.ones((1,m)), np.dot(Theta1, X.T), axis=0)))[1:,]

		Theta1_grad = np.dot(delta2, X)
		Theta2_grad = np.dot(delta3, np.append(np.ones((1,m)), sigmoid(np.dot(Theta1, X.T)), axis=0).T)


		Theta1_grad = Theta1_grad/m + (lam * np.append(np.zeros((Theta1.shape[0],1)), Theta1[:,1:], axis=1))/m
		Theta2_grad = Theta2_grad/m + (lam * np.append(np.zeros((Theta2.shape[0],1)), Theta2[:,1:], axis=1))/m

		grad = np.append(Theta1_grad.flatten('F'), Theta2_grad.flatten('F'))

	if returnType == '':
		return [J, grad]
	elif returnType == 'J':
		return J
	elif returnType == 'grad':
		return grad