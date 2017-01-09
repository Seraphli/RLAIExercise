from GridWorld import GridWorld, RandomPolicy

# Position [-1, -1] means terminal state

print('Figure 4.1')
# World configuration
world_size = (4, 4)
special_state = [([0, 0], [-1, -1], range(4)), ([3, 3], [-1, -1], range(4))]

# Initialize grid world
world = GridWorld(world_size, special_state)

# Policy configuration
policy = RandomPolicy(1)

# Iteration
iteration = 0
diffs = []
while not (world.diff < 1e-4):
    world.step(policy)
    iteration += 1
    diffs.append(world.diff)

# Show value matrix
world.show_value(3, 5)

print('Assume that the transitions from the original states are unchanged.')
# World configuration
world_size = (5, 4)
special_state = [([0, 0], [-1, -1], range(4)), ([3, 3], [-1, -1], range(4)), ([4, 0], [-1, -1], range(4)),
                 ([4, 2], [-1, -1], range(4)), ([4, 3], [-1, -1], range(4)), ([4, 1], [3, 1], [0]),
                 ([4, 1], [4, 1], [1]), ([4, 1], [3, 0], [2]), ([4, 1], [3, 2], [3]), ([3, 1], [3, 1], [1])]

# Initialize grid world
world = GridWorld(world_size, special_state)

# Policy configuration
policy = RandomPolicy(1)

# Iteration
iteration = 0
diffs = []
while not (world.diff < 1e-4):
    world.step(policy)
    iteration += 1
    diffs.append(world.diff)

# Show value matrix
world.show_value(3, 5)

print('Now suppose the dynamics of state 13 are also changed, such that action down from state 13 takes the agent to the new state 15.')
# World configuration
world_size = (5, 4)
special_state = [([0, 0], [-1, -1], range(4)), ([3, 3], [-1, -1], range(4)), ([4, 0], [-1, -1], range(4)),
                 ([4, 2], [-1, -1], range(4)), ([4, 3], [-1, -1], range(4)), ([4, 1], [3, 1], [0]),
                 ([4, 1], [4, 1], [1]), ([4, 1], [3, 0], [2]), ([4, 1], [3, 2], [3])]

# Initialize grid world
world = GridWorld(world_size, special_state)

# Policy configuration
policy = RandomPolicy(1)

# Iteration
iteration = 0
diffs = []
while not (world.diff < 1e-4):
    world.step(policy)
    iteration += 1
    diffs.append(world.diff)

# Show value matrix
world.show_value(3, 5)