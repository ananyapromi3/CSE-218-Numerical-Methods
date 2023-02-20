import matplotlib.pyplot as plt
import numpy as np
import csv
import math


# ------------------------------------------------------------------------------------------------
# Read File
# ------------------------------------------------------------------------------------------------


def ReadTextFile(fileName):
    i = 0
    file = open(fileName, "r")
    for line in file:
        l = line.split()
        if (l[0] and i > 0):
            arrX.append(float(l[0]))
            arrY.append(float(l[1]))
        i += 1
    file.close()


def ReadCxvFile(fileName):
    with open(fileName, 'r') as file:
        csvreader = csv.reader(file)
        # print(csvreader)
        i = 0
        for row in csvreader:
            # print(row)
            # if (row[0].isnumeric()):
            if i > 0:
                arrX.append(float(row[0]))
                arrY.append(float(row[1]))
                # arrY2.append(float(row[2]))
            i = i+1


# ------------------------------------------------------------------------------------------------
# Saturation Growth Regression
# ------------------------------------------------------------------------------------------------


def linearRegression():
    print(arrX)
    print(arrY)
    n = len(arrX)
    recX = 1 / (arrX ** 2)
    recY = 1 / arrY
    print(recX)
    print(recY)
    avgX = np.average(recX)
    avgY = np.average(recY)
    a1 = (sum(recX * recY) - n * avgX * avgY) / \
        (sum(recX * recX) - n * avgX * avgX)
    a0 = avgY - a1 * avgX
    a = 1 / a0
    b = a * a1
    return a, b


def linearFunc(x):
    return (a * (x ** 2)) / (b + x ** 2)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

# File reading
arrX = [0.5, 0.8, 1.5, 2.5, 4.0]
arrY = [1.1, 2.4, 5.3, 7.6, 8.9]
# ReadTextFile("data.txt")
# ReadCxvFile("data.csv")
arrX = np.array(arrX)
arrY = np.array(arrY)
# File check
print(arrX)
print(arrY)
l = min(arrX)
h = max(arrX)

# Plot graph
plt.close('all')
plt.grid()
plt.scatter(arrX, arrY, marker='.', color="black", label='data points')


a, b = linearRegression()
print(a, b)
print(arrX)
print(arrY)
X = np.linspace(l, h, 1000)
funcX = linearFunc(X)
plt.plot(X, funcX, label='y=ax^2/(b+x^2)')

plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
