import matplotlib.pyplot as plt
import numpy as np
from helper import *

from sklearn.decomposition import PCA
x, y = classification_data()

plot_data3d(x, y, "data3d.png")

pca = PCA(n_components=2)
pca.fit(x)
x_transformed = pca.transform(x)

plot_data2d(x_transformed, y, "data2d.png")