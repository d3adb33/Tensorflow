import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def classification_data():
    points=30
    dim=2
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