from __future__ import print_function, division
from Environment import *
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import itertools
import time


def parallel_function(arg):
    n, alpha = arg
    env = RandomWalk(19)
    td = n_step_TD(env)
    return td.learn(100, n, alpha)


def run_experiment(p, arg):
    n, alpha = arg
    print("%d-step TD alpha=%.2f" % (n, alpha))
    args = [arg] * 100
    errors = list(p.map(parallel_function, args))
    error = sum(errors) / 100
    return error


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    true_value = calculate_true_value(19)
    p = Pool(12)
    ns = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.01, 0.01)

    args = list(itertools.product(ns, alphas))
    errors = np.zeros((len(ns), len(alphas)))
    for arg in args:
        errors[np.where(ns == arg[0]), np.where(alphas == arg[1])] = run_experiment(p, arg)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    truncateValue = 0.55
    errors[errors > truncateValue] = truncateValue

    with open("errors.pkl", "w+") as f:
        pickle.dump(errors, f)
    plt.figure()
    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(ns[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig("figure.png")
    plt.show()
