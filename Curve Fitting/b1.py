import matplotlib.pyplot as plt
import numpy as np

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
    recY = arrX * arrY
    A = np.zeros((ord + 1, ord + 1))
    B = np.zeros((ord + 1, 1))
    for i in range(ord + 1):
        for j in range(ord + 1):
            A[i][j] = np.sum(arrX ** (i + j))
    for i in range(ord + 1):
        B[i][0] = np.sum(recY * (arrX ** (i)))
    ai = GaussianElimination(A, B)
    return ai


def poliFunc(x):
    y = 0
    for i in range(len(ai)):
        y += (ai[i] * (x ** i)) / x
    return y


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
# File reading
arrX = [1.0, 2.0, 3.0, 4.0, 5.0]
arrY = [2.2, 2.8, 3.6, 4.5, 5.5]
# ReadTextFile("data.txt")
# ReadCxvFile("data.csv")
arrX = np.array(arrX)
arrY = np.array(arrY)
# File check
# print(arrX)
# print(arrY)
l = min(arrX)
h = max(arrX)
n = len(arrX)
err = np.zeros(n)

# Plot graph
plt.close('all')
plt.grid()
plt.scatter(arrX, arrY, marker='o', color="black", label='data points')

# Polinomial
ai = polinomialRegression(2)
# print('Polinomial:', ai)
X = np.linspace(l-0.5, h+1, 1000)
funcX = poliFunc(X)
print('a =', ai[1][0])
print('b =', ai[2][0])
print('c =', ai[0][0])
x1 = 2.5
x2 = 5.5
y1 = poliFunc(x1)[0]
y2 = poliFunc(x2)[0]
print('At x = 2.5, y =', y1)
print('At x = 5.5, y =', y2)
plt.scatter(x1, y1, marker='o', color='red', label='at x=2.5')
plt.scatter(x2, y2, marker='o', color='green', label='at x=5.5')
# MSE
# err = (poliFunc(arrX) - arrY) ** 2
# print('Variance:', np.average(err))
plt.plot(X, funcX, label='y=a+bx+c/x')


plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
