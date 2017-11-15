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


x1 = [-1,-1]
x2 = [1,1]
w = [1,1]
a = pred(x1,w,0)
b = pred(x2,w,0)
print(a,b)

w = [10,10]
a = pred(x1,w,0)
b = pred(x2,w,0)
print(a,b)