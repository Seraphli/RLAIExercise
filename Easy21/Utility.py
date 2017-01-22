import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def Plot3D(get_value):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(0, 10, 1)
    Y = np.arange(0, 21, 1)
    X, Y = np.meshgrid(X, Y)

    Z = get_value(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def Plot2D(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()
