import numpy as np
import matplotlib.pyplot as plt


# Information
L = 450


def f(x):
    # Equation
    # f(x) = -x**5 + 2*(L**2)*(x**3) + (L**4)*x
    return -5*(x**4) + 6*(L**2)*(x**2) + (L**4)


def plotGraph():
    plt.close('all')
    plt.title("Graph for Visual Representation")
    plt.axhline(y=0, color='c')
    plt.axvline(x=0, color='c')
    x = np.linspace(-600, 600, 1000)
    plt.plot(x, f(x), color='b', markersize=10, label='dy/dx')
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("x (cm)")
    plt.ylabel("y")
    plt.show()


def bisection(lowerBound, upperBound, approxError, maxIteration):
    m = (lowerBound + upperBound)/2
    m_ = m
    error = None
    if f(lowerBound) * f(upperBound) < 0:
        for i in range(maxIteration+1):
            m = (lowerBound + upperBound)/2
            f_ub = f(upperBound)
            f_lb = f(lowerBound)
            f_m = f(m)
            if f_lb * f_m < 0:
                upperBound = m
            elif f_lb * f_m > 0:
                lowerBound = m
            else:
                return m
            if i > 0:
                if (m == 0):
                    print("DIVISION BY ZERO OCCURED!")
                    return None
                error = abs((m - m_) / m) * 100
                print(str(i) + ": x =", m, " Error =  " + str(error) + "%")
                if error <= approxError:
                    return m
            m_ = m
    return None


# Main
plotGraph()
print("\nBisection Method:")
lb = float(input("Estimate lower bound: "))
ub = float(input("Estimate upper bound: "))
max = int(input("Enter max allowed iteration: "))
ans = bisection(lb, ub, 0.5, max)
if ans == None:
    print("Could not calculate")
else:
    print("Answer:\nx = ", ans, "meter")
print()
