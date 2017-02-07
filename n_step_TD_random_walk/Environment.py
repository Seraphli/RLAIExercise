from __future__ import division
import numpy as np

LEFT = 0
RIGHT = 1
LEFT_REWARD = -1
RIGHT_REWARD = 1
GAMMA = 1


def calculate_true_value(state_size):
    true_value = np.zeros(state_size + 2)
    new_value = np.zeros(state_size + 2)
    error = 1
    while error > 10e-12:
        true_value = new_value
        new_value = np.zeros(state_size + 2)
        for i in range(state_size + 2):
            if i == 0:
                new_value[i] = 0
                continue
            if i == state_size + 1:
                new_value[i] = 0
                continue
            if i == 1:
                new_value[i] = 0.5 * (LEFT_REWARD + true_value[i + 1])
                continue
            if i == state_size:
                new_value[i] = 0.5 * (true_value[i - 1] + RIGHT_REWARD)
                continue
            new_value[i] = 0.5 * (true_value[i - 1] + true_value[i + 1])
        error = sum(abs(new_value - true_value))
    return true_value


TRUE_VALUE = calculate_true_value(19)


class RandomWalk(object):
    def __init__(self, state_size):
        self.state_size = state_size
        self.true_value = TRUE_VALUE

    def start(self):
        # state = [0, 1, 2, ... state_size, state_size+1]
        self.state = self.state_size // 2 + 1

    def step(self, choice):
        # choice = np.random.randint(1)
        if choice == LEFT:
            self.state -= 1
        else:
            self.state += 1

        reward = 0
        if self.state == 0:
            reward = LEFT_REWARD
        if self.state == self.state_size + 1:
            reward = RIGHT_REWARD

        return self.state, reward

    def is_terminate(self):
        if self.state in [0, self.state_size + 1]:
            return True

        return False


class n_step_TD(object):
    def __init__(self, environment):
        self.env = environment
        self.value = np.zeros(self.env.state_size + 2)

    def learn(self, episodes, n, alpha):
        error = 0
        for _ in range(episodes):
            self.env.start()
            T = 10 ** 9
            t = 0
            rewards = [0]
            states = [self.env.state]
            while True:
                t += 1
                if t < T:
                    choice = np.random.randint(2)
                    state, reward = self.env.step(choice)
                    rewards.append(reward)
                    states.append(state)
                    if self.env.is_terminate():
                        T = t
                tau = t - n
                if tau >= 0:
                    gammas = np.power(GAMMA, np.arange(tau + 1, min(tau + n, T) + 1))
                    rs = np.array(rewards[tau + 1: min(tau + n, T) + 1])
                    G = np.dot(gammas, rs)
                    # G = np.power(GAMMA, np.arange(tau + 1, min(tau + n, T) + 1)) * np.transpose(
                    #     rewards[tau + 1: min(tau + n, T) + 1])
                    if tau + n < T:
                        G += pow(GAMMA, n) * self.value[states[tau + n]]
                    self.value[states[tau]] += alpha * (G - self.value[states[tau]])
                if tau == T - 1:
                    break

            error += np.sqrt(np.sum(np.power(self.value - self.env.true_value, 2)) / self.env.state_size)
        return error / episodes


if __name__ == '__main__':
    env = RandomWalk(19)
    td = n_step_TD(env)
    td.learn(10, 4, 0.5)
