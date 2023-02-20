import numpy as np


def det(A, flag):
    n = len(A)
    ans = 1
    for i in range(n):
        ans = ans * A[i][i]
    return ans * flag


def forwardElimination(A, B, pivot, showAll):
    row, col = np.shape(A)
    flag = 1
    for i in range(row - 1):
        if pivot:
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
                print("DIVISION BY ZERO ERROR!!!")
                return
            temp = A[j][i] / A[i][i]
            for k in range(col):
                A[j][k] = A[j][k] - temp * A[i][k]
            B[j] = B[j] - temp * B[i]
            if showAll:
                print("Matrix A:")
                print(A)
                print("Matrix B:")
                print(B)
                print()
    if showAll:
        print("Determinant:", "{0:.4}".format(det(A, flag)))


def backSubstitution(A, B):
    row, col = np.shape(A)
    C = np.empty((row, 1))
    if A[row - 1][col - 1] == 0:
        print("DIVISION BY ZERO ERROR!!!")
        return C
    C[row - 1] = B[row - 1] / A[row - 1][col - 1]
    for i in range(row - 2, -1, -1):
        sum = 0
        for j in range(i + 1, row):
            sum += A[i][j] * C[j][0]
        if A[i][i] == 0:
            print("DIVISION BY ZERO ERROR!!!")
            return C
        C[i][0] = (B[i]-sum) / A[i][i]
    return C


def GaussianElimination(A, B, pivot=True, showAll=True):
    forwardElimination(A, B, pivot, showAll)
    return backSubstitution(A, B)


n = int(input())
A = np.empty((n, n))
B = np.empty((n, 1))
for i in range(n):
    rowStr = input()
    row = rowStr.split()
    for j in range(n):
        A[i][j] = float(row[j])
for i in range(n):
    B[i][0] = float(input())

ans = GaussianElimination(A, B, True, False)
print("Answer:")
for i in range(len(ans)):
    print("{0:.4f}".format(ans[i][0]))


'''
Sample Inputs:

3
20 15 10
-3 -2.249 7
5 1 3
45
1.751
9

3
25 5 1
64 8 1
144 12 1
106.8
177.2
279.2

3
0 5 6
4 5 7
9 2 3
11 
16
15
'''
