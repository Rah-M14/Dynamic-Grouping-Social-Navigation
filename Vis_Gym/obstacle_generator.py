import numpy as np
import cv2
from heapq import heappop, heappush

class Pathfinding:
    def __init__(self, gridworld):
        """
        Initialize the Pathfinding class with the gridworld and an empty dictionary for occupied steps.
        """
        self.gridworld = gridworld
        self.occupied_time_steps = {}
        self.obstacle_occupied_points = []

    def is_valid_move(self, pos):
        """
        Check if the position is within grid bounds and is a free space (value 0).
        """
        rows, cols = self.gridworld.shape
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols and self.gridworld[pos[0], pos[1]] == 0

    def is_time_step_valid(self, neighbor, next_step):
        """
        Check if the position is not occupied at the current time step.
        """
        return neighbor not in self.occupied_time_steps or self.occupied_time_steps[neighbor] > next_step

    def heuristic(self, a, b):
        """
        Calculate the Manhattan distance heuristic between points a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_pathfinding(self, start, end):
        """
        Perform A* to find a path from start to end, avoiding time-step collisions.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        open_set = []
        heappush(open_set, (0 + self.heuristic(start, end), 0, start, [start]))
        g_score = {start: 0}
        visited = set([start])

        while open_set:
            _, current_g, current, path = heappop(open_set)
            if current == end:
                return path

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                tentative_g_score = current_g + 1
                if self.is_valid_move(neighbor) and neighbor not in visited:
                    next_step = len(path)
                    if self.is_time_step_valid(neighbor, next_step) and (neighbor not in g_score or tentative_g_score < g_score[neighbor]):
                        visited.add(neighbor)
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, end)
                        heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))

        return []

    def generate_paths(self, quantity, episode_length = 1000):
        """
        Generate paths for all obstacles and update the occupied steps to avoid collisions.
        Ensure the total length of each path is equal to the episode_length.
        """
        def find_free_space(gridworld, occupied_points):
            """
            Find a free space in the gridworld that is not already occupied.
            """
            free_spaces = np.argwhere(gridworld == 0)
            available_spaces = [tuple(pt) for pt in free_spaces if tuple(pt) not in occupied_points]
            return available_spaces[np.random.choice(len(available_spaces))]

        obstacle_start_end_coordinate_list = {}
        occupied_points = []
        

        # Generate unique start and end points for each obstacle
        for x in range(quantity):
            start = find_free_space(self.gridworld, occupied_points)
            occupied_points.append(start)
            end = find_free_space(self.gridworld, occupied_points)
            occupied_points.append(end)
            obstacle_start_end_coordinate_list[str(x + 1)] = {"start_point": start, "end_point": end}

        obstacle_paths = {}
        

        # Generate paths for each obstacle
        for key, coords in obstacle_start_end_coordinate_list.items():
            start, end = tuple(coords["start_point"]), tuple(coords["end_point"])
            path = self.a_star_pathfinding(start, end)
            total_path = path[:]

            # Ensure the total path length is equal to the episode_length
            while len(total_path) < episode_length:
                
                start = total_path[-1]
                end = find_free_space(self.gridworld, occupied_points)
                new_path = self.a_star_pathfinding(start, end)
                if not new_path:
                    break
                total_path.extend(new_path[1:])
                occupied_points.append(end)

            # Truncate the path if it exceeds the episode_length (path truncation)
            obstacle_paths[key] = total_path[:episode_length]
            #adding the new truncated end coordinate of the obstacle
            occupied_points.append(obstacle_paths[key][-1])

            # Update the occupied steps with time step at which it was occupied
            for step, point in enumerate(total_path[:episode_length]):
                self.occupied_time_steps[point] = step
                
            
        # to remove any end points that might have been freed during path truncation  
        occupied_points = [point for point in occupied_points if point in self.occupied_time_steps.keys()]

        
        #storing the obstacles start and end coordinates in the object        
        self.obstacle_occupied_points = occupied_points;

        return obstacle_paths
