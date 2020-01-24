import csv
import numpy as np

filename = 'Admission_Predict.csv'


# reading csv file and input into array
def readcsv(filename):
    ifile = open(filename, "r")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0
    a = []

    for row in reader:
        a.append(row)
        rownum += 1

    ifile.close()
    return a


x = readcsv(filename)

print("hello world")