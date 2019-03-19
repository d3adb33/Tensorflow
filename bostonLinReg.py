import tensorflow as tf
import numpy as np
from keras.datasets import boston_housing

#Dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
train_size = x_train.shape[0]
test_size = x_test.shape[0]

from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
y_pred = regr.predict(x_test)

def compute_error_mse(y, y_p):
    y_p_line = [yp_i for yp_i in y_p]
    y_line = [y_i for y_i in y]
    N  = len(y_line)
    err = (1 / N) * sum([y_p_line[i] - y_line[i]**2 for i in range(N)])
    return err

mse = compute_error_mse(y_test, y_pred)
print("Score: ", score)
print("MSE: ", mse)
