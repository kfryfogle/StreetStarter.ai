import random

import numpy as np
import gymnasium as gym
from tqdm import tqdm

import matrix_mdp

from GametoNumpy import PyGametoNumpy


class Qlearning:
    def __init__(self, width, height, buildings):
        # Hyperparameters
        self.epsilon = 0.9
        self.gamma = 0.9
        self.episodes = 1000
        self.steps_per_episode = 100

        # Mapping of the game
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4
        self.actions = ['up', 'down', 'left', 'right']
        self.buildings = buildings
        self.current_building = buildings[0]
        gtnp = PyGametoNumpy(height, width, buildings)
        self.map = gtnp.convert_to_numpy()
        # self.reward_grid = gtnp.create_rewards(buildings)
        self.reward_grid = gtnp.create_reward_first_step(buildings)

        # Create P_0 for starting state distribution
        self.P_0 = np.array([0 for _ in range(self.num_states)])
        self.P_0[self.current_building.get_position()[0] - 1 + self.current_building.get_position()[1] *
                 self.width] = 1
        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.create_transition_matrix()
        print(self.T)
        # # initialize the Q-table values to 0
        # self.q_table = np.zeros([self.width * self.height, 4])
        # # initialize number of updates for each state-action pair to 0
        # self.num_updates = np.zeros([self.width * self.height, 4])
        self.env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, r=self.reward_grid, p=self.T)
        # self.observation, self.info = self.env.reset(

    def valid_neighbours(self, i, j):
        neighbours = {}
        if i > 0:
            neighbours[0] = (i - 1, j)
        if i < self.width - 1:
            neighbours[1] = (i + 1, j)
        if j > 0:
            neighbours[2] = (i, j - 1)
        if j < self.height - 1:
            neighbours[3] = (i, j + 1)
        return neighbours

    def create_transition_matrix(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.map[x][y] != 0:
                    self.T[x, y, :] = 0
                else:
                    neighbors = self.valid_neighbours(x, y)
                    for action in range(self.num_actions):
                        if action in neighbors:
                            # print(str(neighbors[action][0] * self.width + neighbors[action][1]) + "\n_____")
                            # print(str(neighbors[action][1]) + "\n_____")
                            self.T[neighbors[action][0] * self.width + neighbors[action][
                                1], x * self.width + y, action] = 1
                    # for action in range(4):
                    #     new_y, new_x = y, x
                    #     if action == 0:  # Up
                    #         new_y -= 1
                    #     elif action == 1:  # Right
                    #         new_x += 1
                    #     elif action == 2:  # Down
                    #         new_y += 1
                    #     elif action == 3:  # Left
                    #         new_x -= 1
                    #
                    #     if 0 <= new_y < self.height and 0 <= new_x < self.width:
                    #         self.T[x, y, action] = 1
                    #     else:
                    #         self.T[x, y, action] = 0

    def is_action_valid(self, current_state, action):
        transition_probs = self.T[:, current_state, action]
        if np.any(transition_probs > 0):
            return True
        else:
            return False

    def train(self, num_episodes):
        Q = np.zeros((self.num_states, self.num_actions))
        num_updates = np.zeros((self.num_states, self.num_actions))

        gamma = 0.9
        epsilon = 0.9

        observation, info = self.env.reset()
        optimal_policy = np.zeros(self.num_states)

        for i in tqdm(range(num_episodes)):
            observation, info = self.env.reset()

            while True:
                explr_explo = random.choices([1, 0], weights=[epsilon, 1 - epsilon], k=1)
                if explr_explo[0] == 1:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    action = np.argmax(Q[observation])

                while not self.is_action_valid(observation, action):
                    action = np.random.choice(np.arange(self.num_actions))

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                eta = 1 / (1 + num_updates[observation, action])
                best_next_action = np.argmax(Q[next_observation])
                Q[observation, action] += eta * (
                        reward + gamma * Q[next_observation, best_next_action] - Q[observation, action])
                num_updates[observation, action] += 1
                observation = next_observation

                if terminated or truncated:
                    break

            epsilon *= 0.9999

        optimal_policy = np.argmax(Q, axis=1)
        return optimal_policy
