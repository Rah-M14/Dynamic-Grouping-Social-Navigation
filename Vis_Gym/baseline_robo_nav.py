import gymnasium as gym
from gymnasium import spaces
import numpy as np
import obstacle_generator as obs
from stable_baselines3 import PPO

# Robot Path planning algo
'''
Prerequisite:
(1) A gridworld
(2) Start position e.g. [825,1571]
(3) End position e.g.[835,1561]
(4) don't change the name of the planner function and its input parameters. rest can be changed.
Return Path like this:
Path: [(825, 1571), (826, 1570), (827, 1569), (828, 1568), (829, 1567), (830, 1566), (831, 1565), (832, 1564), (833, 1563), (834, 1562), (835, 1561)]
'''
import numpy as np

class RLEnvironment(gym.Env):
    def __init__(self, gridworld, obstacle_paths, occupied_time_steps, obstacle_occupied_points, number_of_obstacles, episode_length):
        super(RLEnvironment, self).__init__()
        self.gridworld = gridworld
        self.obstacle_paths = obstacle_paths
        self.occupied_time_steps = occupied_time_steps
        self.obstacle_occupied_points = obstacle_occupied_points
        self.robot_position = None
        self.goal_position = None
        self.current_step = 0
        self.global_obstacle_position = []
        self.number_of_obstacles = number_of_obstacles
        self.episode_length = episode_length
        self.robot_path = []  # List to store the robot's path
        
        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # 9 possible actions including no movement
        self.observation_space = spaces.Dict({
            'robot_position': spaces.Box(low=0, high=max(gridworld.shape), shape=(2,), dtype=np.int32),
            'obstacles_positions': spaces.Box(low=0, high=max(gridworld.shape), shape=(number_of_obstacles, 2), dtype=np.int32)
        })
        
        self._initialize_robot_and_goal()
        
       
    def render(self):
        #instead of rendering at each time step we can see animation of the final episode.
        pass
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize the pathfinding class
        pathfinder = obs.Pathfinding(self.gridworld)

        # Generate new paths for obstacles again
        self.obstacle_paths = pathfinder.generate_paths(self.number_of_obstacles, self.episode_length)

        #contains a dictionary of occupied coordinate as keys at different time step as value
        'only one object can occupy a coordinate at a particular time stamp'
        self.occupied_time_steps = pathfinder.occupied_time_steps

        #contains a list of start or end coordinates of each dynamic obstacle
        self.obstacle_occupied_points = pathfinder.obstacle_occupied_points
        
        self.robot_position = None
        self.goal_position = None
        self.current_step = 0
        self.global_obstacle_position = []
        self.robot_path = []  # Reset the robot path
        #making a new path for the robot
        self._initialize_robot_and_goal()
        
        return self._get_obs(),{}

    def _initialize_robot_and_goal(self):
        # Find a free space for the robot start point
        self.robot_position = self._find_free_space()
        # Find a free space for the goal point
        self.goal_position = self._find_free_space()
        self.robot_path.append(self.robot_position) 

    def _find_free_space(self):
        free_spaces = np.argwhere(self.gridworld == 0)
        while True:
            candidate = tuple(free_spaces[np.random.choice(len(free_spaces))])
            if self._is_free_space(candidate):
                return candidate

    def _is_free_space(self, position):
        # Check if the position is occupied by any obstacle at the current time step
        if position in self.occupied_time_steps and self.occupied_time_steps[position] == self.current_step:
            return False
        # Check if the position is occupied by the robot or goal
        if position == self.robot_position or position == self.goal_position:
            return False
        return True

    def step(self, action):
        """
        Perform an action and update the environment state.
        
        :param action: One of the 8 possible actions (0-7).
        :return: Tuple (new_position, obstacles_positions, collision, goal_reached)
        """
        
#         print(f"action taken: {action}")
        
        old_position = self.robot_position
        new_position = self._move(self.robot_position, action)
        collision = not self._is_valid_move(new_position)
        goal_reached = self._check_goal_reached(new_position)
        
        
        if not collision:
            self.robot_position = new_position
            # Update obstacles positions for the current time step
            obstacles_positions = self._get_obstacles_positions()
            self.global_obstacle_position = obstacles_positions

            # Increment the current step
            self.current_step += 1
            self.robot_path.append(new_position)  # Store the new position in the path
            
        else:
            
            #updating the environment with old values if there is collision
            new_position = old_position
            obstacles_positions =  self.global_obstacle_position
            
        obs = self._get_obs()
        reward = self._get_reward(goal_reached, collision)
        terminated = goal_reached
        truncated = self.current_step >= self.episode_length
        done = goal_reached or self.current_step >= self.episode_length
        
        # Print statements for debugging and tracking
        print(f"Step: {self.current_step}, Action: {action}, Position: {self.robot_position}, Collision: {collision}, Goal Reached: {goal_reached}")
        print(f"Obstacle Positions: {obstacles_positions}")
            
            
#         else:
#             obstacles_positions = self._get_obstacles_positions()  # Return current obstacle positions for consistency
       
        return obs, reward, terminated, truncated, {"collision": collision, "goal_reached": goal_reached}

    def _move(self, position, action):
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Right, Down, Left, Up
            (-1, 1), (-1, -1), (1, 1), (1, -1),(0,0)  # Top-Right, Top-Left, Bottom-Right, Bottom-Left, No movement
        ]
        direction = directions[action]
        new_position = (position[0] + direction[0], position[1] + direction[1])
        #return new_position if self._is_valid_move(new_position) else position
        return new_position

    def _is_valid_move(self, position):
        """
        Check if the position is within grid bounds, is a free space (value 0), and is not occupied by any obstacle.
        """
#         print(f"-----------checking if valid move---{position}-------")
        rows, cols = self.gridworld.shape
        if not (0 <= position[0] < rows and 0 <= position[1] < cols):
#             print("going outside the world")
            return False
        if self.gridworld[position[0], position[1]] != 0:
#             print("Collison with the wall")
            return False
        if position in self.occupied_time_steps and self.occupied_time_steps[position] == self.current_step:
#             print("collision with an obstacle")
            return False
        return True

    def _check_goal_reached(self, position):
        return position == self.goal_position

    def _get_obstacles_positions(self):
        obstacles_positions = {}
        for key, path in self.obstacle_paths.items():
            if self.current_step < len(path):
                obstacles_positions[key] = path[self.current_step]
        return obstacles_positions
    
    def _get_obs(self):
        # Create observation dictionary
        obstacles_positions = [self.obstacle_paths[key][self.current_step] for key in self.obstacle_paths.keys()]
        return {
            'robot_position': np.array(self.robot_position, dtype=np.int32),
            'obstacles_positions': np.array(obstacles_positions, dtype=np.int32)
        }
    
    def _get_reward(self, goal_reached, collision):
        if goal_reached:
            return 100  # Reward for reaching the goal
        if collision:
            return -10  # Penalty for collision
        return -1  # Small penalty for each step taken

    

