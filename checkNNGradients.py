def checkNNGradients(lam=0):

	import numpy as np
	from debugInitializeWeights import debugInitializeWeights
	from nnCostFunctionVec import nnCostFunctionVec
	from computeNumericalGradient import computeNumericalGradient

	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m = 5

	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

	X = debugInitializeWeights(m, input_layer_size -1)
	y = np.mod(np.arange(m)+1, num_labels).T

	nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis=0)

	costFunc = lambda p : nnCostFunctionVec(p, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

	[cost, grad] = costFunc(nn_params)

	numgrad = computeNumericalGradient(costFunc, nn_params)

	print(np.append(numgrad.reshape(numgrad.size,1), grad.reshape(grad.size,1), axis=1))
	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
	print(diff)
