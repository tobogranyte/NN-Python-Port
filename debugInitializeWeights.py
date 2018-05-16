def debugInitializeWeights(fan_out, fan_in):

	import numpy as np

	W = np.zeros((fan_out, 1 + fan_in))

	W = np.sin(np.arange(W.size)+1).reshape(W.shape)/10

	return W