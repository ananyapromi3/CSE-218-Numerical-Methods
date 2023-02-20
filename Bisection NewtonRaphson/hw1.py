import numpy as np
import matplotlib.pyplot as plt


# Information
y_b = 0.55
R = .06


def f(x):
    # Equation
    # f(x) = (4/3) * pi * R^3 * p_b - pi * x^2 * (R-x/3) * p_w = 0
    return (x ** 3) - 3 * R * x * x + 4 * (R ** 3) * y_b


def f1(x):
    return 3 * x * x - 6 * R * x


def plotGraph():
    plt.close('all')
    plt.title("Graph for Visual Representation")
    plt.axhline(y=0, color='c')
    plt.axvline(x=0, color='c')
    x = np.linspace(-0.05, 0.2, 1000)
    plt.plot(x, f(x), color='b', markersize=10, label='y=f(x)')
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("x (meter)")
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
                error = abs((m - m_) / m) * 100
                print(str(i) + ": x =", m, " Error =  " + str(error) + "%")
                if error <= approxError:
                    return m
            m_ = m
    return None


def newtonRaphson(initialGuess, approxError, maxIteration):
    guess = initialGuess
    guess_ = guess
    error = None
    for i in range(maxIteration):
        guess = guess_ - f(guess_) / f1(guess_)
        if i > 0:
            error = abs((guess - guess_) / guess) * 100
            print(str(i) + ": x =", guess, " Error =  " + str(error) + "%")
            if error <= approxError:
                return guess
        guess_ = guess
    return None


# Main
plotGraph()
print("\nBisection Method:")
lb = float(input("Estimate lower bound: "))
ub = float(input("Estimate upper bound: "))
max = int(input("Enter max allowed iteration: "))
ans = bisection(lb, ub, 0.005, max)
# ans = bisection(0.03, 0.1, 0.005, 100)
if ans == None:
    print("Could not calculate")
else:
    print("Answer:\nx = ", ans, "meter")
print()
print("\nNewton-Raphson Method:")
r = float(input("Guess root: "))
max = int(input("Enter max allowed iteration: "))
ans = newtonRaphson(r, 0.005, max)
# ans = newtonRaphson(0.03, 0.005, 100)
if ans == None:
    print("Could not calculate")
else:
    print("Answer:\nx = ", ans, "meter")
print()
