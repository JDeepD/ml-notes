import numpy as np
import matplotlib.pyplot as plt
import random

def gen_data(r_min: int = -2, r_max: int = 2, rng: int = 20):
    x = np.arange(rng)
    y = np.arange(rng)
    for i in range(len(y)):
        # Generate a random rv and add that to y[i]
        # to get random scatter plot
        rv = random.randint(r_min,r_max)
        y[i] = y[i] + rv
    
    return x, y

def hypothesis(theta0, theta1, x):
    return theta0 + theta1*x


def gradient_descent(X: np.ndarray, Y:np.ndarray, epoch: int = 1000, alpha: float = 0.0001):
    theta0 = 0
    theta1 = 0
    m = len(Y)
    for it in range(epoch):
        theta0 = theta0 - (alpha/m)*np.sum(hypothesis(theta0, theta1, X) - Y)
        theta1 = theta1 - (alpha/m)*np.sum((hypothesis(theta0, theta1, X) - Y)*X)

    return theta0, theta1

X, Y = gen_data(r_min = -10, r_max = 10, rng=100)
plt.scatter(X, Y)

t0, t1 = gradient_descent(X, Y)
print(t0, t1)
plt.axline((0, t0), slope = t1)
