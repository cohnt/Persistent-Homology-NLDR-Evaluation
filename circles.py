import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_points = 500
Y, color = datasets.make_circles(n_points, random_state=1, factor=0.5, noise=0.05)

fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
# plt.show()

import dionysus as d

max_epsilon = 0.4
k_skeleton = 3

f_1 = d.fill_rips(Y, k_skeleton, max_epsilon)
p_1 = d.homology_persistence(f_1)
dgms_1 = d.init_diagrams(p_1, f_1)
# # d.plot.plot_diagram(dgms[0], show = True)
# # d.plot.plot_diagram(dgms[1], show = True)
# d.plot.plot_bars(dgms[0], show = True)
# d.plot.plot_bars(dgms[1], show = True)

f_2 = d.fill_rips(Y, k_skeleton, max_epsilon)
p_2 = d.homology_persistence(f_2)
dgms_2 = d.init_diagrams(p_2, f_2)

for which_diagram in [0, 1]:
	fig, axes = plt.subplots(nrows=2, ncols=2)
	d.plot.plot_diagram(dgms_1[which_diagram], ax=axes[0,0], show=False)
	d.plot.plot_diagram(dgms_2[which_diagram], ax=axes[0,1], show=False)
	d.plot.plot_bars(dgms_1[which_diagram], ax=axes[1,0], show=False)
	d.plot.plot_bars(dgms_2[which_diagram], ax=axes[1,1], show=False)
	fig.suptitle("Dimension %d" % which_diagram)

plt.show()
