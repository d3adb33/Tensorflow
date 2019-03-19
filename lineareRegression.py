import matplotlib.pyplot as plt
import numpy as np
from helper import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x, y = regression_data()
print("X: ", x.shape)
print("Y: ", y.shape)

plt.scatter(x, y, color= "blue")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train) #training
score  = linear_model.score(x_test, y_test) #testing
print("Score: ", score) #R2 Score

#parameter
# m * x + b
x1 = min(x)
y1 = linear_model.coef_ * x1 + linear_model.intercept_
x2 = max(x)
y2 = linear_model.coef_ * x2 + linear_model.intercept_

plt.plot([x1, x2], [y1, y2], color = "red")
plt.scatter(x, y, color = 'blue')
plt.show()