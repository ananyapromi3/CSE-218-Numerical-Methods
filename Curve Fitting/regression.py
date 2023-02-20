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
# Linear Regression
# ------------------------------------------------------------------------------------------------


def linearRegression():
    n = len(arrX)
    avgX = np.average(arrX)
    avgY = np.average(arrY)
    a1 = (sum(arrX * arrY) - n * avgX * avgY) / \
        (sum(arrX * arrX) - n * avgX * avgX)
    a0 = avgY - a1 * avgX
    return a0, a1


def linearFunc(x):
    return a0 + a1 * x


# ------------------------------------------------------------------------------------------------
# Exponential Regression
# ------------------------------------------------------------------------------------------------


def expRegression():
    n = len(arrX)
    recX = []
    recY = []
    for i in range(n-1):
        if (arrY[i] > 0):
            recX.append(arrX[i])
            recY.append(arrY[i])
    lnY = np.log(arrY)
    sumX = np.sum(arrX)
    sumY = np.sum(lnY)
    avgX = np.average(arrX)
    avgY = np.average(lnY)
    a1 = (n * np.sum(arrX * lnY) - sumX * sumY) / \
        (n * np.sum(arrX * arrX) - sumX * sumX)
    a0 = avgY - a1 * avgX
    A = np.exp(a0)
    return A, a1


def expFunc(x):
    return a * np.exp(b * x)


# ------------------------------------------------------------------------------------------------
# Polinomial Regression
# ------------------------------------------------------------------------------------------------


def forwardElimination(A, B):
    row, col = np.shape(A)
    flag = 1
    for i in range(row - 1):
        max = abs(A[i][i])
        k = i
        for j in (i + 1, col - 1):
            if abs(A[j][i]) > max:
                max = A[j][i]
                k = j
        if k != i:
            A[[i, k]] = A[[k, i]]
            B[[i, k]] = B[[k, i]]
            flag = flag * -1
        for j in range(i + 1, row):
            if A[i][i] == 0:
                A[i][i] = 0.0001
            temp = A[j][i] / A[i][i]
            for k in range(col):
                A[j][k] = A[j][k] - temp * A[i][k]
            B[j] = B[j] - temp * B[i]


def backSubstitution(A, B):
    row, col = np.shape(A)
    C = np.empty((row, 1))
    if A[row - 1][col - 1] == 0:
        A[row - 1][col - 1] = 0.0001
    C[row - 1] = B[row - 1] / A[row - 1][col - 1]
    for i in range(row - 2, -1, -1):
        sum = 0
        for j in range(i + 1, row):
            sum += A[i][j] * C[j][0]
        if A[i][i] == 0:
            A[i][i] = 0.0001
        C[i][0] = (B[i] - sum) / A[i][i]
    return C


def GaussianElimination(A, B):
    forwardElimination(A, B)
    return backSubstitution(A, B)


def polinomialRegression(ord):
    A = np.zeros((ord + 1, ord + 1))
    B = np.zeros((ord + 1, 1))
    for i in range(ord + 1):
        for j in range(ord + 1):
            A[i][j] = np.sum(arrX ** (i + j))
    for i in range(ord + 1):
        B[i][0] = np.sum(arrY * (arrX ** (i)))
    ai = GaussianElimination(A, B)
    return ai


def poliFunc(x):
    y = 0
    for i in range(len(ai)):
        y += ai[i] * (x ** i)
    return y


# ------------------------------------------------------------------------------------------------
# Power Regression
# ------------------------------------------------------------------------------------------------


def powRegression():
    n = len(arrX)
    recX = np.log(arrX)
    recY = np.log(arrY)
    avgX = np.average(recX)
    avgY = np.average(recY)
    a1 = (sum(recX * recY) - n * avgX * avgY) / \
        (sum(recX * recX) - n * avgX * avgX)
    a0 = avgY - a1 * avgX
    a = np.exp(a0)
    b = a1
    return a, b


def powFunc(x):
    return a * (x ** b)


# ------------------------------------------------------------------------------------------------
# Saturation Growth Regression
# ------------------------------------------------------------------------------------------------


def growthRegression():
    n = len(arrX)
    recX = 1 / arrX
    recY = 1 / arrY
    avgX = np.average(recX)
    avgY = np.average(recY)
    a1 = (sum(recX * recY) - n * avgX * avgY) / \
        (sum(recX * recX) - n * avgX * avgX)
    a0 = avgY - a1 * avgX
    a = 1 / a0
    b = a * a1
    return a, b


def growthFunc(x):
    return (a * x) / (x + b)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

# File reading
arrX = []
arrY = []
ReadTextFile("data.txt")
# ReadCxvFile("data.csv")
arrX = np.array(arrX)
arrY = np.array(arrY)
# File check
print(arrX)
print(arrY)
l = 0
h = 0.25
n = len(arrX)
err = np.zeros(n)

# Plot graph
plt.close('all')
plt.grid()
plt.scatter(arrX, arrY, marker='.', color="black", label='data points')

# Linear
a0, a1 = linearRegression()
print('Linear:', a0, a1)
X = np.linspace(l, h, 1000)
funcX = linearFunc(X)
# MSE
err = (linearFunc(arrX) - arrY) ** 2
print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=ax+b')

# Exponential
a, b = expRegression()
print('Exponential:', a, b)
X = np.linspace(l, h, 1000)
funcX = expFunc(X)
# MSE
err = (expFunc(arrX) - arrY) ** 2
print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=ae^bx')

# Polinomial
ai = polinomialRegression(3)
print('Polinomial:', ai)
X = np.linspace(l, h, 1000)
funcX = poliFunc(X)
# MSE
err = (poliFunc(arrX) - arrY) ** 2
print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=P(x)')

# Power
a, b = powRegression()
print('Power:', a, b)
X = np.linspace(l, h, 1000)
funcX = powFunc(X)
# MSE
err = (powFunc(arrX) - arrY) ** 2
print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=ax^b')

# Growth
a, b = growthRegression()
print('Growth:', a, b)
X = np.linspace(l, h, 1000)
funcX = growthFunc(X)
# MSE
err = (growthFunc(arrX) - arrY) ** 2
print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=ax/(b+x)')


plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
