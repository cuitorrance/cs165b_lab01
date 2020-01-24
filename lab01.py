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

def predict(W, X_i):
    return np.sum(W*X_i)
def Y_Hat(W, X):
    return np.array([predict(W, X[i,:]) for i in np.arange(len(X))])
def L(Y, Y_hat):
    return np.sum((Y - Y_hat)**2)*1/(len(Y))
def e(X, Y, W): 
    Y_hats = Y_Hat(W,X)
    l = L(Y, Y_hats)
    return l
def e_prime(X, Y, W, lam):
    ss_pred_err = e(X,Y,W)
    ss_weights = np.sum(W**2)
    return ss_pred_err + lam*ss_weights

#initialize starting weights as all ones
weights = np.ones(8)

