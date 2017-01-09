from GridWorld import GridWorld, OptimalPolicy
import matplotlib.pyplot as plt

# World configuration
world_size = (5, 5)
special_state = [([0, 1], [4, 1], 10), ([0, 3], [2, 3], 5)]

# Initialize grid world
world = GridWorld(world_size, special_state)

# Policy configuration
policy = OptimalPolicy(0.9)

# Iteration
iteration = 0
diffs = []
while not (world.diff < 1e-4):
    world.step(policy)
    iteration += 1
    diffs.append(world.diff)

# Show value matrix
world.show_value(3)

# Plot gradient curve
plt.figure(1)
plt.plot(range(iteration), diffs)
plt.show()