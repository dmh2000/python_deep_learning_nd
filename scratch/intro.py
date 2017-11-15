import numpy as np


def f(X, w, b):
    return np.dot(X, w) + b


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    e = np.exp(L)
    s = np.sum(e)
    p = np.divide(e, s)
    return p


def pred(X, w, b):
    a = f(X, w, b)
    b = sigmoid(a)
    return b


def xentropy(p):
    n = np.log(p)
    s = np.sum(n * -1.0)
    return s


def cross_entropy(Y, P):
    s = 0
    for i in range(len(Y)):
        v = Y[i] * np.log(P[i]) + (1.0 - Y[i]) * np.log(1.0 - P[i])
        s += v
    return -s


def E(X, W, y):
    s = 0
    x = f(X, W)
    x = sigmoid(x)
    for i in range(len(y)):
        e1 = (1.0 - y[i]) * np.log(1.0 - x)
        e2 = y[i] * np.log(x)
        s = e1 + e2

    m = 1.0 / len(y)
    e = -1.0 * m * s
    return e


## ======================================== tests
# def test():
#    w = [4, 5, -9]
#    X = [[1, 1, 1], [2, 4, 1], [5, -5, 1], [-4, 5, 1]]
#
#    for x in X:
#        v = f(x, w)
#        s = sigmoid(v)
#        print(x, v, s)
#
#
# print(xentropy([0.8, 0.7, 0.9]))

# print(cross_entropy([1, 1, 0], [0.8, 0.7, 0.9]))


w = [[2.0, 6.0], [3.0, 5.0], [5.0, 4.0]]
b = [-2, -2.2, -3]
x = [0.4, 0.6]

print(pred(x, w[0], b[0]))
print(pred(x, w[1], b[1]))
print(pred(x, w[2], b[2]))
