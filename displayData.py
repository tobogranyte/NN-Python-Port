def displayData(X):

	import matplotlib.pyplot as plt
	import numpy as np

	(m, n) = X.shape

	# compute rows and columns
	i16 = np.dtype(np.int16)
	display_rows = np.floor(np.sqrt(m)).astype(i16)
	display_cols = np.ceil(m/display_rows).astype(i16)

	example_width = np.round(np.sqrt(n)).astype(i16)
	example_height = (n / example_width).astype(i16)

	pad = 1

	display_array = np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
	curr_ex = 0
	for j in range(0, display_rows):
		for i in range(0, display_cols):
			if curr_ex == m:
				break

			max_val = np.amax(np.absolute(X[curr_ex]))
			row_start = pad + j * (example_height + pad)
			col_start = pad + i * (example_width + pad)
			display_array[row_start:(row_start + example_height), col_start:(col_start + example_width)] = X[curr_ex].reshape(example_height, example_width, order=	'F') / max_val
			curr_ex = curr_ex + 1

		if curr_ex == m:
			break

	plt.axis("off")
	plt.gray()
	plt.imshow(display_array, vmin = -1, vmax = 1)
	plt.show()
	return
