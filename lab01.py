import csv
import numpy as np
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

#input dataset
dataset = load_csv('Admission_Predict.csv')

#delete first row
del dataset[0]

#convert string to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)

#get epxected values
expected = np.zeros(400)
i = 0
for row in dataset:
    expected[i] = row[8]
    i = i +1

#get parameters
for row in dataset:
    #set first column to bias = 1
    row[0] = 1
    del row[8]

def predict(X_i, W):
    return np.sum(W*X_i)
def Y_Hat(X, W):
    return np.array([predict(X[i,:],W) for i in np.arange(len(X))])
def L(Y, Y_hat):
    return np.sum((Y - Y_hat)**2)*1/(len(Y))
def e(X, Y, W): 
    Y_hats = Y_Hat(X,W)
    l = L(Y, Y_hats)
    return l
def e_prime(X, Y, W, lam):
    ss_pred_err = e(X,Y,W)
    ss_weights = np.sum(W**2)
    return ss_pred_err + lam*ss_weights

def SGD_run_epoch(X, Y, W, lam, alpha):
    n = len(X)    
    indices = np.arange(0, n)
    rand_indices = np.random.choice(indices, n, replace = False)
    
    yhat = Y_Hat(X,W)

    for i in rand_indices:
        W[i] = W[i] - ( alpha * (2(yhat[i] - Y[i]) * X[i]) )

    return W

# Calculate root mean squared error
def mse(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return mean_error

#training phase
def train(x, y, alpha, lam, nepoch, param, epsilon):
    for i in range(nepoch):
        param = SGD_run_epoch(x, y, param, lam, alpha)
        if e_prime(x, y, param, lam, alpha) < epsilon:
            return param
    

# #validation phase
# def validation(x,y, param):

# #testing phase
# def test(x, param):


#run for all epochs
def SGDSolver(x, y, alpha = 10, lam = 10, nepoch = 100, epsilon = 0.05, param = [1, 1, 1, 1, 1,1,1]):
    #add bias to weight
    param.append(1)

    param = train(x, y, alpha, lam, nepoch, param, epsilon)

    validation(x,y, param)

    test(x,param)









        

    




