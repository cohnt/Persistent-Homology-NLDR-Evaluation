import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_points = 600
X, color = datasets.make_swiss_roll(n_points)
n_neighbors = 6
n_components = 2

method = manifold.Isomap(n_neighbors, n_components)
Y = method.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
plt.show()