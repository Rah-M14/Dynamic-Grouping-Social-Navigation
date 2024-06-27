#custom files
import gridworld_converter
import robo_nav_algo as robo
import free_space_finder
import utils
import obstacle_generator as obs
import baseline_robo_nav as baseline

#standard files
# %matplotlib notebook
import cv2

# Load the Image
image = cv2.imread('example6.png', cv2.IMREAD_COLOR)
# 2d matrix representation of the above blueprint
# value of scale can be used to control how much you want to zoom into the grid world
gridworld = gridworld_converter.grid_convert('example6.png',scale = 3)
#print(f"Number of rows: {gridworld.shape[0]}\nNumber of columns: {gridworld.shape[1]}")

number_of_obstacles = 30
episode_length = 10000
total_timesteps = 10000000  # Total training steps

# Initialize the pathfinding class
pathfinder = obs.Pathfinding(gridworld)
# Generate paths for obstacles
obstacle_paths = pathfinder.generate_paths(number_of_obstacles, episode_length)
#contains a dictionary of occupied coordinate as keys at different time step as value
'only one object can occupy a coordinate at a particular time stamp'
occupied_time_steps = pathfinder.occupied_time_steps
#contains a list of start or end coordinates of each dynamic obstacle
obstacle_occupied_points = pathfinder.obstacle_occupied_points
print("Starting the engine")
'''
Help:
You can see the obstacle path by obstacle_paths['1'] or obstacle_paths['2']
'''
import os
import wandb

import obstacle_generator as observ

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import heapq
import numpy as np
from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output

class DGSN_Env(gym.Env):
    def __init__(self, gridworld, obstacle_paths, occupied_time_steps, obstacle_occupied_points, number_of_obstacles, episode_length):
        super(DGSN_Env, self).__init__()
        print("Initialised")
        #print(f"It begins!")
        self.gridworld = gridworld
        self.grid_size = gridworld.shape
        
        self.obstacle_paths = obstacle_paths
        self.occupied_time_steps = occupied_time_steps
        self.obstacle_occupied_points = obstacle_occupied_points
        self.episode_length = episode_length
        self.boundary_threshold = 0
        self.max_dynamic_obstacles = number_of_obstacles
        self.num_static_obstacles = 0
        self.max_static_obstacles = 0
        self.number_of_obstacles = number_of_obstacles
        self.num_dynamic_obstacles = number_of_obstacles
        self.robot_position = None
        self.goal_position = None
        self.agent_pos = self.robot_position
        self.goal_pos = self.goal_position
        self.ep_no = 0
        self.global_obstacle_position = []
        self.close_call, self.discomfort, self.current_step, self.ep_no = 0, 0, 1, 0
        self.dist_factor = 4
        self.thresh = 1.5
        self.max_steps = episode_length
        self.total_reward = 0
        self.robot_path = []

        self.action_space = spaces.Discrete(9)  # 9 possible actions including no movement
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(low=0, high=max(self.grid_size), shape=(2,), dtype=np.int32),
            'dyn_obs': spaces.Box(low=0, high=max(self.grid_size), shape=(number_of_obstacles, 2), dtype=np.int32)
        })
        
        self._initialize_robot_and_goal()
        self.dynamic_obstacles = self._init_dynamic_obstacles(obstacle_paths)
        
        wandb.login()
        wandb.init(project='DGSN_Grid_World')
        
        self.safe = True
        
        # Evaluation Metrics
        self.Avg_Success_Rate = 0
        self.Avg_Collision_Rate, self.ep_collision_rate, self.collision_count = 0, 0, 0
        self.Avg_Min_Time_To_Collision, self.min_time_to_collision = 0, 0
        self.Avg_Wall_Collision_Rate, self.ep_wall_collision_rate, self.wall_collision_count = 0, 0, 0
        self.Avg_Obstacle_Collision_Rate, self.ep_obstacle_collision_rate, self.obstacle_collision_count = 0, 0, 0
        self.Avg_Human_Collision_Rate, self.ep_human_collision_rate, self.human_collision_count = 0, 0, 0
        self.Avg_Timeout = 0
        self.Avg_Path_Length = 0
        self.Avg_Stalled_Time, self.stalled_time = 0, 0
        self.Avg_Group_Inhibition_Rate, self.ep_group_inhibition_rate, self.group_inhibition = 0, 0, 0
        self.Avg_Discomfort, self.ep_discomfort = 0, 0
        self.Avg_Human_Distance, self.ep_human_distance, self.human_distance = 0, 0, 0
        self.Avg_Closest_Human_Distance, self.closest_human_distance = 0, 0
        self.Min_Closest_Human_Distance = 0
        self.goal_reached = 0
        self.timeout = 0
        
    def _initialize_robot_and_goal(self):
        print("initialize_robot_and_goal")
        # Find a free space for the robot start point
        self.robot_position = self._find_free_space()
        self.agent_pos = self.robot_position
        # Find a free space for the goal point
        self.goal_position = self._find_free_space()
        self.goal_pos = self.goal_position
        self.robot_path.append(self.robot_position) 
        
    def _find_free_space(self):
        print("_find_free_space")
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

    def _init_dynamic_obstacles(self, obstacle_paths):
        obstacles = []
        for i in range(self.number_of_obstacles):
            start_pos = obstacle_paths[str(i+1)][0]
            end_pos = obstacle_paths[str(i+1)][-1]
            path = obstacle_paths[str(i+1)]
            distance = np.linalg.norm(np.array(self.agent_pos) - np.array(start_pos))
            obstacles.append({
                'start': start_pos,
                'end': end_pos,
                'current': start_pos,
                'angle': np.rad2deg(np.arctan2(self.agent_pos[1] - start_pos[1], self.agent_pos[0] - start_pos[0])).astype('float'),
                'distance': int(distance),
                'path': path if path else []
            })
        return obstacles
    
    def _check_goal_reached(self, position):
        return position == self.goal_position
    
    def _is_valid_move(self, position):
        """
        Check if the position is within grid bounds, is a free space (value 0), and is not occupied by any obstacle.
        """
    #         #print(f"-----------checking if valid move---{position}-------")
        rows, cols = self.gridworld.shape
        if not (0 <= position[0] < rows and 0 <= position[1] < cols):
    #             #print("going outside the world")
            return False
        if self.gridworld[position[0], position[1]] != 0:
    #             #print("Collison with the wall")
            return False
        if position in self.occupied_time_steps and self.occupied_time_steps[position] == self.current_step:
    #             #print("collision with an obstacle")
            return False
        return True
    
    def _compute_path(self, start, end):
        def heuristic(a, b):
            return np.linalg.norm(a - b)

        def a_star(start, goal):
            open_set = []
            heapq.heappush(open_set, (0, tuple(start)))
            came_from = {}
            g_score = {tuple(start): 0}
            f_score = {tuple(start): heuristic(start, goal)}

            while open_set:
                _, current = heapq.heappop(open_set)
                current = np.array(current)

                if np.array_equal(current, goal):
                    path = []
                    while tuple(current) in came_from:
                        path.append(current)
                        current = came_from[tuple(current)]
                    path.append(start)
                    path.reverse()
                    return path
                
                val = 1
                neighbors = [current + [val, 0], current + [-val, 0], current + [0, val], current + [0, -val]]
                neighbors = [np.clip(neighbor, 0, self.grid_size - 0.5) for neighbor in neighbors]

                for neighbor in neighbors:
                    tentative_g_score = g_score[tuple(current)] + heuristic(current, neighbor)
                    if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g_score
                        f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[tuple(neighbor)], tuple(neighbor)))
            return []

        return a_star(start, end)
    
    def _get_obstacles_positions(self):
        obstacles_positions = {}
        for key, path in self.obstacle_paths.items():
            if self.current_step < len(path):
                obstacles_positions[key] = path[self.current_step]
        return obstacles_positions
    
    def _move(self, position, action):
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Right, Down, Left, Up
            (-1, 1), (-1, -1), (1, 1), (1, -1),(0,0)  # Top-Right, Top-Left, Bottom-Right, Bottom-Left, No movement
        ]
        direction = directions[action]
        new_position = (position[0] + direction[0], position[1] + direction[1])
        return new_position
    
    def step(self, action):
        previous_agent_pos = self.robot_position
        new_position = self._move(self.robot_position, action)
        collision = not self._is_valid_move(new_position)
        goal_reached = self._check_goal_reached(new_position)
        
        if not collision:
            self.robot_position = new_position
            self.agent_pos = new_position
            obstacles_positions = self._get_obstacles_positions()
            self.global_obstacle_position = obstacles_positions
            self.current_step += 1
            self.robot_path.append(new_position)
        else:
            new_position = previous_agent_pos
            obstacles_positions = self.global_obstacle_position
            
        obs = self._get_obs()
        reward = self._compute_reward(previous_agent_pos)
        done = goal_reached or self.current_step >= self.episode_length
        truncated = self.current_step >= self.episode_length
        
        #print(f"Agent's Current Position {previous_agent_pos}")
        #print(f"Goal Position : {self.goal_pos}")
        #print(f"Agent's New Position {self.agent_pos}")
        #print(f"Steps taken : {self.current_step}")
        
        for i, obstacle in enumerate(self.dynamic_obstacles):
            if len(obstacle['path']) > 1:
                obstacle['current'] = obstacles_positions[str(i+1)]
        
        #print(f"#####################################")
        #print(f"Reward Obtained : {reward}")
        #print(f"#####################################")
        
        self.total_reward += reward
        wandb.log({"Reward": reward, "Total_Reward": self.total_reward})
        
        self.done = self._is_done()
        if self.current_step >= self.max_steps:
            self.done = True
            reward -= 2500
            self.timeout += 1
            wandb.log({"TimeOut": self.timeout})
        
        return obs, reward, self.done, truncated, {}

    def _compute_reward(self, previous_agent_pos):
        reward_c = 0
        if self.safe:
            self.min_time_to_collision += 1
        previous_dist_to_goal = np.linalg.norm(np.array(previous_agent_pos) - np.array(self.goal_pos))
        current_dist_to_goal = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        del_distance = current_dist_to_goal - previous_dist_to_goal

        #print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
        #print(f"Distance to the Goal : {current_dist_to_goal}")
        #print(f"Del_Distance : {del_distance}")

        if del_distance > 0:
            reward_c -= del_distance * self.dist_factor
        elif del_distance == 0:
            reward_c = -2
        else:
            reward_c += (-del_distance) * self.dist_factor * 2

        #print(f"Reward from Del_Dist : {reward_c}")

        for i, obs in enumerate(self.dynamic_obstacles):
            distance = np.linalg.norm(np.array(obs['current']) - np.array(self.agent_pos))
            self.human_distance += distance
            #print(f"******************************")
            #print(f"Obstacle #{i} Distance : {distance}")

            if distance < 3 and distance != 0:  
                self.closest_human_distance = min(self.closest_human_distance, distance)
                self.close_call += 1
                pen_1 = (10 / distance) * 2
                reward_c -= pen_1  
                #print(f"Penalty Obtained : {pen_1}")

        #print(f"Reward Post Dynamic Manouevres : {reward_c}")
        #print(f"******************************")

        grouped_obstacles = self._group_dynamic_obstacles()
        for j, group in enumerate(grouped_obstacles):
            group_center = np.mean(group, axis=0)
            if np.linalg.norm(np.array(self.agent_pos) - group_center) < self.thresh:                
                reward_c -= 50
                self.group_inhibition += 1
                self.close_call += 1
                #print(f"Oops Inhibited the Group {j} - Penalty -50")
                wandb.log({"Group_Inhibition": self.group_inhibition})

        if any(np.array_equal(self.agent_pos, np.array(obs['current'])) for obs in self.dynamic_obstacles):
            self.collision_count += 1
            self.human_collision_count += 1
            self.safe = False
            reward_c -= 30  
            #print(f"Collided with a Human!!!")
            #print(f"Post Human Collision Reward : {reward_c}")

        if np.any(np.array(self.agent_pos) <= self.boundary_threshold) or np.any(np.array(self.agent_pos) >= np.array(self.grid_size) - self.boundary_threshold):
            reward_c -= 15  
            #print(f"Pretty Close to the Boundary!!!")
            #print(f"Post Boundary Penalty Reward : {reward_c}")

        if np.any(np.array(self.agent_pos) == 0) or np.any(np.array(self.agent_pos) >= np.array(self.grid_size)):
            self.collision_count += 1
            self.wall_collision_count += 1
            self.safe = False
            reward_c -= 20  
            #print(f"Collided with the wall!!!")
            #print(f"Post Wall Collision Reward : {reward_c}")

        reward_c -= 1

        if self._is_done():
            reward_c += 3000
            #print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
            #print(f"Goal Reached!!!!!")
            #print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
            self.goal_reached += 1
            wandb.log({'Goal_Reached': self.goal_reached})

        return reward_c

    
    def _is_done(self):
        return np.array_equal(np.array(self.agent_pos), np.array(self.goal_pos))
    
    def _get_obs(self):
        agent_state = np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.int32)
        
        dynamic_obstacle_states = np.array(
            [(np.rad2deg(np.arctan2(self.agent_pos[1] - ob['current'][1], self.agent_pos[0] - ob['current'][0])), 
              np.linalg.norm(np.array(self.agent_pos) - np.array(ob['current']))) 
             for ob in self.dynamic_obstacles] + 
            [np.zeros(2, dtype=np.int32) for _ in range(self.max_dynamic_obstacles - len(self.dynamic_obstacles))], 
            dtype=np.int32)

        return {
            'agent': agent_state,
            'dyn_obs': dynamic_obstacle_states
        }
    

        
    def _group_dynamic_obstacles(self):
        positions = np.array([np.array(obs['current']) for obs in self.dynamic_obstacles])
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(positions)
        labels = clustering.labels_

        grouped_obstacles = []
        for label in set(labels):
            if label != -1:
                group = positions[labels == label]
                grouped_obstacles.append(group)
        return grouped_obstacles
    
    def render(self, mode='human'):
        pass
    
    def reset(self, **kwargs):
        
        wandb.log({"Episode": self.ep_no})
        self.ep_no += 1
        print(f"PRINTING & LOGGING!!! {self.ep_no}")

        if self.current_step > 0:
            self.ep_human_distance = self.human_distance / (self.num_dynamic_obstacles * self.current_step)
            self.ep_discomfort = self.close_call / self.current_step
            self.ep_collision_rate = self.collision_count / self.current_step
            self.ep_wall_collision_rate = self.wall_collision_count / self.current_step
            self.ep_obstacle_collision_rate = self.obstacle_collision_count / self.current_step
            self.ep_human_collision_rate = self.human_collision_count / self.current_step
            self.ep_group_inhibition_rate = self.group_inhibition / self.current_step
        else:
            self.ep_human_distance = 0
            self.ep_discomfort = 0
            self.ep_collision_rate = 0
            self.ep_wall_collision_rate = 0
            self.ep_obstacle_collision_rate = 0
            self.ep_human_collision_rate = 0
            self.ep_group_inhibition_rate = 0

        self.Avg_Collision_Rate = ((self.ep_no - 1) * self.Avg_Collision_Rate + self.ep_collision_rate) / self.ep_no
        self.Avg_Min_Time_To_Collision = ((self.ep_no - 1) * self.Avg_Min_Time_To_Collision + self.min_time_to_collision) / self.ep_no
        self.Avg_Wall_Collision_Rate = ((self.ep_no - 1) * self.Avg_Wall_Collision_Rate + self.ep_wall_collision_rate) / self.ep_no
        self.Avg_Obstacle_Collision_Rate = ((self.ep_no - 1) * self.Avg_Obstacle_Collision_Rate + self.ep_obstacle_collision_rate) / self.ep_no
        self.Avg_Human_Collision_Rate = ((self.ep_no - 1) * self.Avg_Human_Collision_Rate + self.ep_human_collision_rate) / self.ep_no
        self.Avg_Path_Length = ((self.ep_no - 1) * self.Avg_Path_Length + self.current_step) / self.ep_no
        self.Avg_Stalled_Time = ((self.ep_no - 1) * self.Avg_Stalled_Time + self.stalled_time) / self.ep_no
        self.Avg_Discomfort = ((self.ep_no - 1) * self.Avg_Discomfort + self.ep_discomfort) / self.ep_no
        self.Avg_Human_Distance = ((self.ep_no - 1) * self.Avg_Human_Distance + self.ep_human_distance) / self.ep_no
        self.Avg_Closest_Human_Distance = ((self.ep_no - 1) * self.Avg_Closest_Human_Distance + self.closest_human_distance) / self.ep_no
        self.Avg_Group_Inhibition_Rate = ((self.ep_no - 1) * self.Avg_Group_Inhibition_Rate + self.ep_group_inhibition_rate) / self.ep_no
        
        wandb.log({
            "Ep_Total_Reward": self.total_reward, 
            "Ep_Collision_Count": self.collision_count, "Ep_Min_Time_To_Collision": self.min_time_to_collision, 
            "Ep_Wall_Collision_Count": self.wall_collision_count, "Ep_Obstacle_Collision_Count": self.obstacle_collision_count, 
            "Ep_Human_Collision_Count": self.human_collision_count, "Ep_Collision_Rate": self.ep_collision_rate,
            "Ep_Wall_Collision_Rate": self.ep_wall_collision_rate, "Ep_Obstacle_Collision_Rate": self.ep_obstacle_collision_rate,
            "Ep_Human_Collision_Rate": self.ep_human_collision_rate, "Ep_Path_Length": self.current_step,
            "Ep_Stalled_Time": self.stalled_time, "Ep_Discomfort": self.ep_discomfort,
            "Ep_Avg_Human_Distance": self.ep_human_distance, "Ep_Closest_Human_Distance": self.closest_human_distance,
            "Ep_Close_Calls": self.close_call, "Ep_Group_Inhibition": self.ep_group_inhibition_rate,
            "Avg_Collision_Rate": self.Avg_Collision_Rate, "Avg_Min_Time_To_Collision": self.Avg_Min_Time_To_Collision,
            "Avg_Wall_Collision_Rate": self.Avg_Wall_Collision_Rate, "Avg_Obstacle_Collision_Rate": self.Avg_Obstacle_Collision_Rate,
            "Avg_Human_Collision_Rate": self.Avg_Human_Collision_Rate, "Avg_Path_Length": self.Avg_Path_Length,
            "Avg_Stalled_Time": self.Avg_Stalled_Time, "Avg_Discomfort": self.Avg_Discomfort,
            "Avg_Human_Distance": self.Avg_Human_Distance, "Avg_Closest_Human_Distance": self.Avg_Closest_Human_Distance,
            "Avg_Group_Inhibition_Rate": self.Avg_Group_Inhibition_Rate
        })
        
        #print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print(f"RUN DETAILS!!! \n")
        #print(f"Ep_Total_Reward : {self.total_reward} \n"), 
        #print(f"Ep_Collision_Count : {self.collision_count}")
        #print(f"Ep_Wall_Collision_Count : {self.wall_collision_count}")
        #print(f"Ep_Obstacle_Collision_Count : {self.obstacle_collision_count}")
        #print(f"Ep_Human_Collision_Count : {self.human_collision_count}")
        #print(f"Ep_Min_Time_To_Collision : {self.min_time_to_collision}")
        #print(f"Ep_Stalled_Time : {self.stalled_time}")
        #print(f"Ep_Avg_Human_Distance : {self.Avg_Human_Distance}")
        
        #print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print("Creating the new Episode")
        
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
        
        pathfinder = observ.Pathfinding(self.gridworld)
        self.obstacle_paths = pathfinder.generate_paths(self.number_of_obstacles, self.episode_length)
        self.occupied_time_steps = pathfinder.occupied_time_steps
        self.obstacle_occupied_points = pathfinder.obstacle_occupied_points
        self.robot_position = None
        self.goal_position = None
        self.current_step = 0
        self.global_obstacle_position = []
        self.robot_path = []
        self._initialize_robot_and_goal()
        
        self.agent_pos = self.robot_position
        self.goal_pos = self.goal_position
        self.num_static_obstacles = 0
        self.num_dynamic_obstacles = len(self.obstacle_paths)
        
        self.dynamic_obstacles = self._init_dynamic_obstacles(self.obstacle_paths)
        self.safe = True
        self.done = False
        
        self.Avg_Success_Rate = 0
        self.ep_collision_rate, self.collision_count = 0, 0
        self.min_time_to_collision = 0
        self.ep_wall_collision_rate, self.wall_collision_count = 0, 0
        self.ep_obstacle_collision_rate, self.obstacle_collision_count = 0, 0
        self.ep_human_collision_rate, self.human_collision_count = 0, 0
        self.Avg_Timeout = 0
        self.Avg_Path_Length = 0
        self.stalled_time = 0
        self.ep_group_inhibition_rate, self.group_inhibition = 0, 0
        self.ep_discomfort = 0
        self.ep_human_distance, self.human_distance = 0, 0
        self.closest_human_distance = 0
        self.Min_Closest_Human_Distance = 0
        self.close_call, self.discomfort, self.current_step = 0, 0, 1
        self.total_reward = 0
        
        #print("*******************************************************************\n")
        #print("Initialized the environment with the following")
        #print("Agent's Initial Position :", self.agent_pos)
        #print("Goal Position :", self.goal_pos)
        #print("Number of Dynamic Obstacles :", self.num_dynamic_obstacles)
        #print("Dynamic Obstacle theta & dist :", self.dynamic_obstacles, "\n")
        #print("*******************************************************************")
        
        return self._get_obs(), {}

class CustomCallback(BaseCallback):
    def __init__(self, env, render_freq=1, save_freq=100, save_path=None, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{self.n_calls}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {save_file}")
        
        return True

print("Ready")
env = DGSN_Env(gridworld, obstacle_paths, occupied_time_steps, obstacle_occupied_points, number_of_obstacles, episode_length)
obs = env.reset()

log_path = os.path.join('Train','Logs')
save_path = os.path.join('Train', 'Saved_Models', 'PPO_Grid_1')

combined_callback = CustomCallback(env=env, render_freq=1, save_freq=100, save_path=save_path, verbose=1)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path, device='cuda', n_epochs=1000)

model.learn(total_timesteps=10000000, callback=combined_callback)
