import matplotlib.pyplot as plt
import numpy as np


# Read file
arrX = []
arrY = []
file = open("datapoints.txt", "r")
for line in file:
    l = line.split()
    if (l[0] and l[0].isnumeric()):
        arrX.append(float(l[0]))
        arrY.append(float(l[1]))
file.close()
arrX = np.array(arrX)
arrY = np.array(arrY)
# File check
# print(a323rrY)


def interval(arrX, arrY, val, n):
    xVals = np.zeros(n+1)
    yVals = np.zeros(n + 1)
    l = len(arrX)
    if n+1 > l:
        print("Not enough points")
        return None, None
    if val < arrX[0] or val > arrX[l-1]:
        print("Out of range")
        return None, None
    for i in range(l):
        if arrX[i] == val:
            print("Answer:", arrY[i])
            return None, None
    for i in range(l-1):
        if arrX[i] < val and arrX[i+1] > val:
            p = i
            q = i+1
    k = 0
    while k <= n:
        if p < 0 and q >= l:
            print("Interval selection error")
            return None, None
        elif p >= 0 and np.abs(arrX[p]-val) < np.abs(arrX[q]-val):
            xVals[k] = arrX[p]
            yVals[k] = arrY[p]
            p = p-1
        elif q >= l:
            xVals[k] = arrX[p]
            yVals[k] = arrY[p]
            p = p-1
        else:
            xVals[k] = arrX[q]
            yVals[k] = arrY[q]
            q = q+1
        k = k + 1
    return xVals, yVals


def f_b(xValArr, yValArr):
    n = len(xValArr)
    B = np.zeros((n, n))
    for i in range(n):
        B[i][i] = yValArr[i]
    # print(B)
    for i in range(n):
        for j in range(1, n):
            if i-j >= 0 and i-j+1 < n and xValArr[i] != xValArr[i-j]:
                B[i][i-j] = (B[i][i-j+1]-B[i-1][i-j]) / \
                    (xValArr[i]-xValArr[i-j])
    return B


def Calculate(x, B, xValArr):
    n = len(xValArr)
    mult = np.ones(n+1)
    for i in range(1, n):
        for j in range(i):
            mult[i] = mult[i]*(x-xValArr[j])
    ans = 0
    for i in range(n):
        ans = ans + B[i][0]*mult[i]
    # print("Mult:")
    # print(mult)
    return ans


# Check
# print(arrX)
# print(arrY)
v = float(input())
# v = 45.6
# n = 4
# interval(arrX, arrY, v, n)
# xVals = np.array(xVals)
# yVals = np.array(yVals)
ans = np.zeros(7)
err = np.zeros(7)
for i in range(2, 7):
    xVals, yVals = interval(arrX, arrY, v, i)
    print(xVals)
    print(yVals)
    B = f_b(xVals, yVals)
    ans[i] = Calculate(v, B, xVals)
print(ans)


# xVals, yVals = interval(arrX, arrY, v, n)
# print("X:")
# print(xVals)
# print("Y:")
# print(yVals)
# B = f_b(xVals, yVals)
# print("B:")
# print(B)
# ans = Calculate(v, B, xVals)
# print(ans)
# print(ans)
# print(err)


# Graph plot
# plt.plot(arrX, arrY)
# plt.plot(v, ans, marker="o", mec="red")
plt.grid()
for i in range(len(arrX)):
    plt.plot(arrX[i], arrY[i], marker=".", color="grey")
# plt.plot(v, ans2, marker="o", mec="red", mfc="purple")
# plt.plot(v, ans3, marker="o")
# xx = np.linspace(0, 200, 1000)
# xx = np.linspace(1, 120, 200)
# yy = np.zeros(200)
# for i in range(len(xx)):
#     yy[i] = Calculate(xx[i], B, xVals)
# yy = Calculate(xx, B, xVals)
# plt.plot(xx, yy, color='red', markersize=10, label='y=f(x)')
# plt.show()


# v = float(input())
# Degree 1
n = 1
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[1] = Calculate(v, B, xVals)
# print("For degree 1:", ans)
# xx1 = np.linspace(1, 120, 500)
# yy1 = np.zeros(500)
# for i in range(len(xx1)):
#     yy1[i] = Calculate(xx1[i], B, xVals)
# plt.plot(xx1, yy1, color='red', markersize=10, label='y=f(x)')
# plt.show()

# Degree 2
n = 2
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[2] = Calculate(v, B, xVals)
print("For degree 2:", ans[2])
xx = np.linspace(0, 120, 200)
yy = np.zeros(200)
for i in range(len(xx)):
    yy[i] = Calculate(xx[i], B, xVals)
plt.plot(xx, yy, color='red', markersize=10)
err[2] = (np.abs((ans[2]-ans[1])/ans[2]))*100
print("Error:", err[2], "%")
print("B:")
print(B)

# Degree 3
n = 3
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[3] = Calculate(v, B, xVals)
print("For degree 3:", ans[3])
xx = np.linspace(1, 120, 200)
yy = np.zeros(200)
for i in range(len(xx)):
    yy[i] = Calculate(xx[i], B, xVals)
plt.plot(xx, yy, color='green', markersize=10)
err[3] = (np.abs((ans[3]-ans[2])/ans[3]))*100
print("Error:", err[3], "%")


# Degree 4
n = 4
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[4] = Calculate(v, B, xVals)
print("For degree 4:", ans[4])
xx = np.linspace(1, 120, 200)
yy = np.zeros(200)
for i in range(len(xx)):
    yy[i] = Calculate(xx[i], B, xVals)
plt.plot(xx, yy, color='yellow', markersize=10)
err[4] = (np.abs((ans[4]-ans[3])/ans[4]))*100
print("Error:", err[4], "%")

# Degree 5
n = 5
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[5] = Calculate(v, B, xVals)
print("For degree 5:", ans[5])
xx = np.linspace(1, 120, 200)
yy = np.zeros(200)
for i in range(len(xx)):
    yy[i] = Calculate(xx[i], B, xVals)
plt.plot(xx, yy, color='black', markersize=10)
err[5] = (np.abs((ans[5]-ans[4])/ans[5]))*100
print("Error:", err[5], "%")

# Degree 6
n = 6
xVals, yVals = interval(arrX, arrY, v, n)
B = f_b(xVals, yVals)
ans[6] = Calculate(v, B, xVals)
print("For degree 6:", ans[6])
xx = np.linspace(1, 120, 200)
yy = np.zeros(200)
for i in range(len(xx)):
    yy[i] = Calculate(xx[i], B, xVals)
plt.plot(xx, yy, color='violet', markersize=10)
err[6] = (np.abs((ans[6]-ans[5])/ans[6]))*100
print("Error:", err[6], "%")
# for i in range(len(arrX)):
#     print(B[])

plt.show()


# err = np.zeros(7)
# for i in range(7):
