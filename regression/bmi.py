import numpy as np
import pandas as pd

# TODO: Load the data in Pandas

bmi_life_data = pd.read_csv("../../deep-learning/linear-regression/bmi_and_life_expectancy.csv")

# Print the data
print(bmi_life_data)


import matplotlib.pyplot as plt

x = np.array(bmi_life_data[["BMI"]])
y = np.array(bmi_life_data["Life expectancy"])


def draw_data(x, y):
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color='blue', edgecolor='k')
    plt.xlabel('BMI')
    plt.ylabel('Life expectancy')


def display(m, b, color='g'):
    r = np.arange(min(x), max(x), 0.1)
    plt.plot(r, m*r+b, color)


draw_data(x, y)
plt.show()

epochs = 1000
learning_rate = 0.01


# TODO: Finish the code for this function
def linear_regression(x, y):
    # Initialize m and b
    m = 1
    b = 0

    for epoch in range(epochs):
        # TODO: Use the square trick to update the weights
        # and run it for a number of epochs
        for i in range(len(x)):
            yhat = m * x[i] + b
            df   = y[i] - yhat
            dw1 = m * df * learning_rate / len(x)
            dw2 =     df * learning_rate / len(x)
            m += dw1
            b += dw2

    return (m, b)

m,b = linear_regression(x,y)
draw_data(x, y)
display(m[0], b[0])
plt.show()