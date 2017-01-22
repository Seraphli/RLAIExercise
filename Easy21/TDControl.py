import Easy21
import numpy as np
import random


class Sarsa:
    def __init__(self, q_star):
        self.q_star = q_star

    def solve(self, n_lambda, need_list=False, iters=1000, n0=100):
        self.value = np.zeros((10, 21))
        self.q = np.zeros((10, 21, 2))
        c = np.zeros((10, 21, 2))
        self.Ns = np.zeros((10, 21))
        self.n0 = n0
        if need_list:
            mse = []
        else:
            mse = 0
        for _ in range(iters):
            # Generate an episode
            s = Easy21.new_game()
            a = self.policy(s)
            while not s == Easy21.TERMINAL:
                next_s, r = Easy21.step(s, Easy21.ACTION[a])
                if not next_s == Easy21.TERMINAL:
                    next_a = self.policy(next_s)
                    # Update Q value
                    c[s[0] - 1, s[1] - 1, a] += 1
                    alpha = 1 / c[s[0] - 1, s[1] - 1, a]
                    self.q[s[0] - 1, s[1] - 1, a] += alpha * (
                        r + n_lambda * self.q[next_s[0] - 1, next_s[1] - 1, next_a] - self.q[s[0] - 1, s[1] - 1, a])
                else:
                    next_a = -1
                    c[s[0] - 1, s[1] - 1, a] += 1
                    alpha = 1 / c[s[0] - 1, s[1] - 1, a]
                    self.q[s[0] - 1, s[1] - 1, a] += alpha * (r - self.q[s[0] - 1, s[1] - 1, a])

                s = next_s
                a = next_a
            if need_list:
                mse.append(((self.q - self.q_star) ** 2).mean())
        if not need_list:
            mse = ((self.q - self.q_star) ** 2).mean()
        return mse

    def policy(self, s):
        epsilon = self.n0 / (self.n0 + self.Ns[s[0] - 1, s[1] - 1])
        self.Ns[s[0] - 1, s[1] - 1] += 1
        if random.random() < epsilon:
            a = random.randint(0, 1)
        else:
            a = np.argmax(self.q[s[0] - 1, s[1] - 1, :])

        return a


if __name__ == '__main__':
    import pickle
    from Utility import Plot2D

    with open("q_star_500.obj", 'rb') as output:
        q_star = pickle.load(output)
    sarsa = Sarsa(q_star)
    lambda_mse = []
    for n_lambda in np.arange(0, 1.1, 0.1):
        lambda_mse.append(sarsa.solve(n_lambda, n0=500))
    Plot2D(np.arange(0, 1.1, 0.1), lambda_mse)
