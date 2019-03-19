import matplotlib.pyplot as plt
import numpy as np
from helper import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

x, y = regression_data()
print("X: ", x.shape)
print("Y: ", y.shape)

plt.scatter(x, y, color= "blue")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train) #training
score  = linear_model.score(x_test, y_test) #testing

def compute_error_mae(y_true, y_pred):
    ret_mae = mean_absolute_error(y_true, y_pred)
    return ret_mae

def compute_error_mse(y_true, y_pred):
    ret_mse = mean_squared_error(y_true, y_pred)
    return ret_mse

y_predict = linear_model.predict(x_test)
mae = compute_error_mae(y_test, y_predict)
mse = compute_error_mse(y_test, y_predict)

print("Linear Model Score (MAE): ", mae)
print("Linear Model Score (MSE): ", mse)
print("Linear Model Score (R2): ", score)
