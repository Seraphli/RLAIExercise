from __future__ import print_function
import numpy as np


class Move:
    """
    moves represent direction you can take, clockwise
    """

    def __init__(self):
        self.moves = [0 for _ in range(4)]


class Action(Move):
    def __init__(self):
        Move.__init__(self)


class Reward(Move):
    def __init__(self):
        Move.__init__(self)


class Grid:
    def __init__(self, i, j, world_size):
        self.i = i
        self.j = j
        self.action = Action()
        self.reward = Reward()
        # action up
        self.set_a_r(0, i == 0, [i - 1, j])
        # action down
        self.set_a_r(1, i == world_size[0] - 1, [i + 1, j])
        # action left
        self.set_a_r(2, j == 0, [i, j - 1])
        # action right
        self.set_a_r(3, j == world_size[1] - 1, [i, j + 1])

    def set_a_r(self, direction, condition, next_s):
        if condition:
            self.action.moves[direction] = [self.i, self.j]
            self.reward.moves[direction] = -1.0
        else:
            self.action.moves[direction] = next_s
            self.reward.moves[direction] = 0


class GridWorld:
    def __init__(self, world_size, special_state):
        self.world = []
        self.world_size = world_size
        for i in range(world_size[0]):
            self.world.append([])
            for j in range(world_size[1]):
                self.world[i].append(Grid(i, j, world_size))
        for state in special_state:
            state_a, state_b, reward = state
            for i in range(4):
                self.world[state_a[0]][state_a[1]].action.moves[i] = state_b
                self.world[state_a[0]][state_a[1]].reward.moves[i] = reward
        self.value = np.zeros(world_size)
        self.new_value = np.ones(world_size)
        self.diff = self.diffs()

    def show_value(self, accuracy):
        prefix = ' %' + '%d' % (accuracy + 3) + '.' + '%d' % accuracy + 'f|'
        split_line = '|' + ('-' * (accuracy + 4) + '+') * (self.world_size[1] - 1) + '-' * (accuracy + 4) + '|'
        print(split_line)
        for i in range(self.world_size[0]):
            print('|', end='')
            for j in range(self.world_size[1]):
                print(prefix % self.value[i, j], end='')
            print('')
            print(split_line)
        print('')

    def step(self, policy):
        self.new_value = np.zeros(self.world_size)
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                self.new_value[i, j] = policy.cal_new_value(i, j, self)
        self.diff = self.diffs()
        self.value = self.new_value

    def diffs(self):
        return np.sum(np.abs(self.value - self.new_value))


class Policy:
    def __init__(self, discount):
        self.discount = discount

    def cal_new_value(self, i, j, grids):
        pass


class OptimalPolicy(Policy):
    def cal_new_value(self, i, j, grids):
        value = [0 for _ in range(4)]
        for direction in range(4):
            new_pos = grids.world[i][j].action.moves[direction]
            value[direction] = grids.world[i][j].reward.moves[direction] + self.discount * grids.value[
                new_pos[0], new_pos[1]]
        return np.max(value)


if __name__ == '__main__':
    pass
