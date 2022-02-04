import matplotlib.pyplot as plt
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu_function(x):
    return np.maximum(x, 0)


def elu_func(x, a):
    return np.where(x > 0, x, a * (np.exp(x) - 1))


x = np.linspace(-6, 6, 200)
sig = sigmoid(x)
relu = relu_function(x)
elu = elu_func(x, 1)

plt.figure()
plt.plot(x, sig)
plt.xlabel("x")
plt.ylabel("sigmoide(x)")

plt.figure()
plt.plot(x, relu)
plt.xlabel("x")
plt.ylabel("ReLu(x)")

plt.figure()
plt.plot(x, elu)
plt.xlabel("x")
plt.ylabel("ELU(x)")

plt.show()
