from __future__ import print_function, division
from Environment import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools


def run_one_episode(args):
    index, n_state, episodes, is_print = args
    if is_print and index % (episodes // 1000) == 0:
        print(index / episodes * 100)
    env = RandomWalk(n_state)
    T = 0
    env.start()
    while not env.is_terminate():
        env.step(np.random.randint(2))
        T += 1
    return T


def count_turns(p, n_state, episodes, is_print):
    args = zip(range(episodes), [n_state] * episodes, [episodes] * episodes, [is_print] * episodes)
    result = list(p.map(run_one_episode, args))
    count = np.zeros((max(result) + 1, 1))
    for r in result:
        count[r] += 1

    count /= episodes
    return count


def display_count(count):
    print(max(count))
    print(np.argmax(count))
    print(min(count))
    print(np.argmin(count))
    index = np.where(count != 0)[0]
    plt.figure()
    plt.plot(index, count[count != 0])
    plt.show()


def display_n_states(n_states, max_count):
    plt.figure()
    plt.plot(n_states, max_count)
    plt.show()


def cal_19_states_figure():
    episodes = 10000000
    count = count_turns(p, 19, episodes, True)
    display_count(count)


def cal_11_states_figure():
    episodes = 10000000
    count = count_turns(p, 11, episodes, True)
    display_count(count)


def cal_n_states_figure():
    episodes = 1000000
    n_states = range(5, 31, 2)
    max_count = []
    for n_state in n_states:
        print(n_state)
        count = count_turns(p, n_state, episodes, False)
        max_count.append(np.argmax(count))
    display_n_states(n_states, max_count)


def parallel_learn(args):
    n_states, true_value, n, alpha = args
    runs = 100
    episodes = 100
    error = 0
    for run in range(runs):
        env = RandomWalk(n_states)
        env.true_value = true_value
        td = n_step_TD(env)
        error+= td.learn(episodes, n, alpha)
    error /= runs
    return error


def display_n_turns(n_states, best_n):
    plt.figure()
    plt.plot(n_states, best_n)
    plt.show()


def cal_n_turns_figure():
    n_states = np.arange(5, 33, 2)
    ns = np.arange(1, 17)
    alphas = np.arange(0, 1.01, 0.01)
    best_n = []
    for n_state in n_states:
        true_value = calculate_true_value(n_state)
        error_n = []
        for n in ns:
            print(n_state, n)
            args = list(itertools.product([n_state], [true_value], [n], alphas))
            error_alpha = list(p.map(parallel_learn, args))
            error_n.append(min(error_alpha))
        best_n.append(ns[error_n.index(min(error_n))])
    display_n_turns(n_states, best_n)


if __name__ == '__main__':
    p = Pool(12)
    cal_n_turns_figure()
