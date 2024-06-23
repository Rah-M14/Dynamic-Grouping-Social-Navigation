# Code for live visualisation:

'''
Prerequisite:
(1) Have a precalculated path through some algo
(2) Have a gridline ready
'''

# %matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def live_animate(cleaned_gridworld, path):

    # Create a figure for the live plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cleaned_gridworld, cmap='gray')

    # Initialize the line for the path
    line, = ax.plot([], [], color='yellow', linewidth=2)

    # Initialize the scatter points for start and end
    start_point = path[0]
    end_point = path[-1]
    ax.scatter([start_point[1]], [start_point[0]], c='green', marker='o', label='Start')
    ax.scatter([end_point[1]], [end_point[0]], c='red', marker='x', label='End')
    ax.legend()

    # Function to update the line for each frame in the animation
    def update_line(num, path, line):
        line.set_data([point[1] for point in path[:num]], [point[0] for point in path[:num]])
        return line,


    # Create the animation
    ani = animation.FuncAnimation(fig, update_line, frames=len(path), fargs=[path, line], interval=1, blit=True)

    # Show the live plot
    plt.title('Live Path Animation')

    


