import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def classification_data():
    points = 40
    dim = 3
    mu_vec1 = np.array([0 for i in range(dim)])
    cov_mat1 = np.identity(dim)
    class1_data = np.random.multivariate_normal(mu_vec1, cov_mat1, int(points/2)).T

    mu_vec2 = np.array([3 for i in range(dim)])
    cov_mat2 = np.identity(dim)
    class2_data = np.random.multivariate_normal(mu_vec2, cov_mat2, int(points/2)).T

    x = np.concatenate((class1_data.T, class2_data.T))
    y = np.array([1 for _ in range(len(class1_data.T))] + [2 for _ in range(len(class2_data.T))])
    return x, y

def regression_data():
    mean = [0, 0]
    cov = [[97, 3], [573, -77]]
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    x = x.reshape(-1, 1)
    return x, y

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# Plotting Functions
def plot_data2d(x, y, file_name="image.png"):
    for xi, yi in zip(x, y):
        if yi == 1:
            plt.plot(xi[0], xi[1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
        else:
            plt.plot(xi[0], xi[1], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.title("Data 2D")
    plt.savefig(file_name)
    plt.show()

def plot_data3d(x, y, file_name="image.png"):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10   
    for xi, yi in zip(x, y):
        if yi == 1:
            ax.scatter(xi[0], xi[1], xi[2], 'o', s=40, color='blue', alpha=0.5, label='class1')
        else:
            ax.scatter(xi[0], xi[1], xi[2], '^', s=40, alpha=0.5, color='red', label='class2')
    plt.title("Data 3D")
    plt.savefig(file_name)
    plt.show()