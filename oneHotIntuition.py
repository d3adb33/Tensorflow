import matplotlib.pyplot as plt
import numpy as np

def generate_data_or():
    x = [[0,0], [1,1], [1,0], [0,1]]
    y = [0, 1, 1, 1]
    return x, y

# y zu oneHotVector umwandeln

def to_one_hot(y, num_classes):
    y_one_hot = np.zeros(shape=(len(y), num_classes)) # 4 x 2
    for i, y_i in enumerate(y):
        #y_oh = [1 if c == y_i else 0 for c in range(num_classes)]
        y_oh = np.zeros(shape=num_classes)
        y_oh[y_i] = 1
        y_one_hot[i] = y_oh
    return y_one_hot

x, y = generate_data_or()
print(y)
y = to_one_hot(y, num_classes = 2)
print(y)