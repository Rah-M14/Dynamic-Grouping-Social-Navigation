import numpy as np

# Function to find a free space in the gridworld
def find(gridworld):
    # Get all coordinates where the value is 0 (free space)
    free_spaces = np.argwhere(gridworld == 0)
    # Randomly select one of the free spaces
  
    return tuple(free_spaces[np.random.choice(free_spaces.shape[0])])
