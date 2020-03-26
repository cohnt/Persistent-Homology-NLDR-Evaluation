import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_points = 500
X, color = datasets.make_swiss_roll(n_points)
n_neighbors = 12
n_components = 2