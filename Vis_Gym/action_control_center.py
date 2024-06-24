# Function to check if a position is within the grid bounds and is free
def is_valid_move(grid, pos):
    rows, cols = grid.shape
    # Check if the position is within bounds and is a free space (value 0)
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols and grid[pos[0], pos[1]] == 0