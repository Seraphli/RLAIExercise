import Easy21
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle


class MonteCarloControl:
    def __init__(self):
        self.value = np.zeros((10, 21))
        self.q = np.zeros((10, 21, 2))
        self.c = np.zeros((10, 21, 2))
        self.N0 = 100
        self.Ns = np.zeros((10, 21))

    def solve(self, iters, n0=100):
        for _ in range(iters):

            # Generate an episode
            history = []
            s = Easy21.new_game()
            while not s == Easy21.TERMINAL:
                epsilon = n0 / (n0 + self.Ns[s[0] - 1, s[1] - 1])
                self.Ns[s[0] - 1, s[1] - 1] += 1
                if random.random() < epsilon:
                    a = random.randint(0, 1)
                else:
                    a = np.argmax(self.q[s[0] - 1, s[1] - 1, :])

                next_s, r = Easy21.step(s, Easy21.ACTION[a])
                history.append((s, a, r))
                s = next_s

            # Update Q value
            G = 0
            for i in range(len(history) - 1, -1, -1):
                s, a, r = history[i]
                G += r
                self.c[s[0] - 1, s[1] - 1, a] += 1
                alpha = 1 / self.c[s[0] - 1, s[1] - 1, a]
                self.q[s[0] - 1, s[1] - 1, a] += alpha * (G - self.q[s[0] - 1, s[1] - 1, a])

    def optimal_policy(self):
        for i in range(10):
            for j in range(21):
                self.value[i, j] = self.q[i, j, np.argmax(self.q[i, j, :])]


if __name__ == '__main__':
    mc = MonteCarloControl()
    mc.solve(10000000, 500)
    mc.optimal_policy()
    with open("value.obj", 'wb') as output:
        pickle.dump(mc.value, output)

    with open("value.obj", 'rb') as output:
        mc.value = pickle.load(output)


    def get_stat_val(x, y):
        return mc.value[x, y]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(0, 10, 1)
    Y = np.arange(0, 21, 1)
    X, Y = np.meshgrid(X, Y)

    Z = get_stat_val(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
