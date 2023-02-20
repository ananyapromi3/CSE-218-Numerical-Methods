import numpy as np
import matplotlib.pyplot as plt

# info
Cme = 5*pow(10, -4)
xInitial = 1.22*pow(10, -4)


def f(x):
    return -(6.73*x+6.725*pow(10, -8)+7.26*pow(10, -4)*Cme)/(3.62*pow(10, -12)*x+3.908*pow(10, -8)*x*Cme)


def TrapezoidRule(a, b, n):
    h = (b-a)/n
    ans = f(a)+f(b)
    for i in range(1, n):
        ans = ans+2*f(a+h*i)
    ans *= (h/2)
    return ans


def SimpsonsRule(a, b, n):
    h = (b-a)/n
    ans = f(a)+f(b)
    for i in range(1, n):
        if i % 2 == 1:
            ans += 4*f(a+i*h)
    for i in range(2, n-1):
        if i % 2 == 0:
            ans += 2*f(a+i*h)
    ans *= (h/3)
    return ans


xi = 0.75*xInitial
xf = .25*xInitial

# Multiple-application trapezoid rule
n = int(input())
print("Multiple-application trapezoid rule")
trapAns = np.zeros(n)
trapErr = np.zeros(n)
for i in range(n):
    trapAns[i] = TrapezoidRule(xi, xf, i+1)
    if i > 0:
        trapErr[i] = abs((trapAns[i]-trapAns[i-1])/trapAns[i])*100
trapErr[0] = None
for i in range(n):
    print("n =", i+1, "\tIntegral Value:",
          trapAns[i], "\tApproximate Relative Error:", trapErr[i], "%")

# Simpsons’ 1/3 rule
print("Simpsons' 1/3 rule")
simpAns = np.zeros(n)
simpErr = np.zeros(n)
for i in range(n):
    simpAns[i] = SimpsonsRule(xi, xf, 2*(i+1))
    if i > 0:
        simpErr[i] = abs((simpAns[i]-simpAns[i-1])/simpAns[i])*100
simpErr[0] = None
for i in range(n):
    print("n =", 2*(i+1), "\tIntegral Value:",
          simpAns[i], "\tApproximate Relative Error:", simpErr[i], "%")


# Plot Graph
n = 7
x = np.array([1.22*pow(10, -4), 1.20*pow(10, -4), 1.0*pow(10, -4),
             0.8*pow(10, -4), 0.6*pow(10, -4), 0.4*pow(10, -4), 0.2*pow(10, -4)])
y = np.zeros(n)
for i in range(n):
    y[i] = SimpsonsRule(x[0], x[i], 10)
plt.plot(x, y, marker="o")
plt.xlabel("Concentration of Oxygen\n(moles/cm^3)")
plt.ylabel("Time\n(seconds)")
plt.grid()
plt.show()
