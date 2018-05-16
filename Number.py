# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import scipy.io as sio
from scipy import optimize
import numpy as np
from displayData import displayData
from nnCostFunctionVec import nnCostFunctionVec
from sigmoidGradient import sigmoidGradient
from checkNNGradients import checkNNGradients
from randInitializeWeights import randInitializeWeights
from predict import predict

#set up the model
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

#load data file
print('Loading and Visualizing Data ...\n')
ex4Data = sio.loadmat('ex4data1.mat')

# create an 8-bit integer type to hold both the image data (8 bits per pixel) and the label data
i16 = np.dtype(np.int16)

X = np.array(ex4Data['X'])

print(X[0,0].dtype)

y = np.array(ex4Data['y'], dtype='int64')

#replace '10' in labels with '0' since unlike matlab and octave, Python is zero indexed
np.place(y, y==10, [0])

m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]

displayData(X[sel,:])

r = np.random.permutation(m)
XRand = np.zeros((X.shape))
yRand = np.zeros((y.shape), dtype = 'int64')
for i in range(m):
	XRand[i,:] = X[r[i],:]
	yRand[i] = y[r[i]]

Xtraining = XRand[0:3500,:]
ytraining = yRand[0:3500]

Xvalidation = XRand[3500:,:]
yvalidation = yRand[3500:]

ex4Thetas = sio.loadmat('ex4weights.mat')

Theta1 = np.array(ex4Thetas['Theta1'])
Theta2 = np.array(ex4Thetas['Theta2'])

nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis = 0)

lam = 1

[J, grad] = nnCostFunctionVec(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

print('Cost at parameters loaded from ex4Weights: ', J, ' \n(this value should be about 0.383770)\n ')

g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]: ", g)

checkNNGradients(lam)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

nn_params = np.append(initial_Theta1.flatten('F'), initial_Theta2.flatten('F'))

costFunc = lambda p : nnCostFunctionVec(p, input_layer_size, hidden_layer_size, num_labels, Xtraining, ytraining, lam, returnType = 'J')
gradFunc = lambda p : nnCostFunctionVec(p, input_layer_size, hidden_layer_size, num_labels, Xtraining, ytraining, lam, returnType = 'grad')

nn_params = optimize.fmin_cg(costFunc, nn_params, fprime=gradFunc, maxiter=500 )

Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1, order = 'F')
Theta2 = nn_params[(hidden_layer_size *(input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size +1), order = 'F')

displayData(Theta1[:, 1:])

pred = predict(Theta1, Theta2, Xtraining)
pred = pred.reshape(pred.size,1)

trainingAccuracy = np.mean(pred == ytraining) * 100

print("Training accuracy: ", trainingAccuracy)

pred = predict(Theta1, Theta2, Xvalidation)
pred = pred.reshape(pred.size,1)

validationAccuracy = np.mean(pred == yvalidation) * 100

print("Validation accuracy: ", validationAccuracy)

