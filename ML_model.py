import numpy as np
import matplotlib as plt
from helper import *

#CLASSIFICATION
# x, y = classification_data()
# plt.scatter(x[:,0], x[:,1], c=y[:])
# plt.show()

# x1 = -2
# y1 = 5
# x2 = 6
# y2 = -3

# plt.plot([x1, x2], [y1, y2], color= "red")
# plt.scatter(x[:,0], x[:,1], c=y[:])
# plt.show()


#REGRESSION

x, y = regression_data()
plt.scatter(x, y, color = "blue")
plt.show()

x1 = -44
y1 = 8
x2 = 55
y2 = -9

plt.plot([x1, x2], [y1, y2], color= "red")
plt.scatter(x, y, c='blue')
plt.show()