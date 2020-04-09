import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_points = 100
Y, color = datasets.make_circles(n_points, random_state=1, factor=0.5, noise=0)

fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
# plt.show()

import dionysus as d

max_epsilon = 1
k_skeleton = 3

f = d.fill_rips(Y, k_skeleton, max_epsilon)
p = d.homology_persistence(f)
dgms = d.init_diagrams(p, f)

fig, axes = plt.subplots(nrows=2, ncols=2)
d.plot.plot_diagram(dgms[0], ax=axes[0,0], show=False)
d.plot.plot_diagram(dgms[1], ax=axes[0,1], show=False)
d.plot.plot_bars(dgms[0], ax=axes[1,0], show=False)
d.plot.plot_bars(dgms[1], ax=axes[1,1], show=False)

axes[0,0].set_title("Dimension 0")
axes[0,1].set_title("Dimension 1")

plt.show()
