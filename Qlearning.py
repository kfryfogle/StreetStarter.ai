import random

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import sys

import constants

np.set_printoptions(threshold=sys.maxsize)

import matrix_mdp

from GametoNumpy import PyGametoNumpy


class Qlearning:
    def __init__(self, width, height, buildings, after_first, paths):
        # Hyperparameters
        self.epsilon = 0.9
        self.gamma = 0.9
        self.episodes = 1000
        self.steps_per_episode = 100
        self.after_first = after_first
        self.paths = paths

        # Mapping of the game
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4
        self.actions = ['up', 'down', 'left', 'right']
        self.buildings = buildings
        if after_first:
            self.current_building = buildings[-1]
        else:
            self.current_building = buildings[0]
        gtnp = PyGametoNumpy(height, width, buildings)
        self.map = gtnp.convert_to_numpy()
        # self.reward_grid = gtnp.create_rewards(buildings)
        if after_first:
            self.reward_grid = gtnp.create_reward_after_first(paths)
        self.reward_grid = gtnp.create_reward_first_step(buildings)
        print(self.map)

        # Create P_0 for starting state distribution
        self.P_0 = np.array([0 for _ in range(self.num_states)])
        self.starting_index = self.current_building.get_position()[0] - 1 + self.current_building.get_position()[
            1] * self.width
        self.P_0[self.starting_index] = 1
        # print(self.P_0)
        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.create_transition_matrix()
        # print(self.reward_grid)
        self.env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, r=self.reward_grid, p=self.T)

    def all_neighbours(self, i, j):
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

    def valid_neighbours(self, i, j):
        neighbours = {}
        if i > 0 and self.map[i - 1, j] == 0:
            neighbours[0] = (i - 1, j)
        if i < self.width - 1 and self.map[i + 1, j] == 0:
            neighbours[1] = (i + 1, j)
        if j > 0 and self.map[i, j - 1] == 0:
            neighbours[2] = (i, j - 1)
        if j < self.height - 1 and self.map[i, j + 1] == 0:
            neighbours[3] = (i, j + 1)
        return neighbours

    def create_transition_matrix(self):
        for x in range(self.width):
            for y in range(self.height):
                neighbors = self.valid_neighbours(x, y)
                if x == 1 and y == 2:
                    print(self.map)
                    print(neighbors)
                for action in range(self.num_actions):
                    if action in neighbors:
                        # print(str(neighbors[action][0] * self.width + neighbors[action][1]) + "\n_____")
                        # print(str(neighbors[action][1]) + "\n_____")
                        self.T[neighbors[action][0] * self.width + neighbors[action][
                            1], x * self.width + y, action] = 1

    def is_action_valid(self, current_state, action):
        transition_probs = self.T[:, current_state, action]
        if np.any(transition_probs > 0):
            return True
        else:
            return False

    def current_coordinates(self, current_state):
        return current_state // self.height, current_state % self.width

    def check_adjacency(self, current_state, visited_buildings):
        x, y = self.current_coordinates(current_state)
        neighbors = self.all_neighbours(x, y)
        visited_coordinates = []
        # get all coordinates of the current visited buildings
        for building in visited_buildings:
            for i in range(building.get_size()[0]):
                for j in range(building.get_size()[1]):
                    visited_coordinates.append((building.get_position()[0] + i, building.get_position()[1] + j))

        for neighbor in neighbors.values():
            if neighbor not in visited_coordinates:
                # check if the neighbor is part of an unvisited building
                for building in self.buildings:
                    building_coordinates = []
                    # get all coordinates of the current building
                    for i in range(building.get_size()[0]):
                        for j in range(building.get_size()[1]):
                            building_coordinates.append(
                                (building.get_position()[0] + i, building.get_position()[1] + j))
                    # if the neighbor is part of the building, add the building to the visited buildings
                    if neighbor in building_coordinates:
                        print("unvisited neighbor found: ", neighbor)
                        visited_buildings.append(building)
                        break
        return visited_buildings

    def print_building_coordinates(self):
        # check if the neighbor is part of an unvisited building
        for building in self.buildings:
            building_coordinates = []
            # get all coordinates of the current building
            for i in range(building.get_size()[0]):
                for j in range(building.get_size()[1]):
                    building_coordinates.append((building.get_position()[0] + i, building.get_position()[1] + j))

    def train(self, num_episodes):
        Q = np.zeros((self.num_states, self.num_actions))
        num_updates = np.zeros((self.num_states, self.num_actions))

        gamma = 0.95
        epsilon = 0.97

        observation, info = self.env.reset()
        optimal_policy = np.zeros(self.num_states)

        for i in tqdm(range(num_episodes)):
            observation, info = self.env.reset()
            visited_buildings = []
            while True:
                # print("observation state, coordinates: ", observation, self.current_coordinates(observation))
                # mark adjacent buildings as visited
                visited_buildings = self.check_adjacency(observation, visited_buildings)
                # if all buildings are visited, break
                if len(visited_buildings) == len(self.buildings):
                    break

                explr_explo = random.choices([1, 0], weights=[epsilon, 1 - epsilon], k=1)
                if explr_explo[0] == 1:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    action = np.argmax(Q[observation])

                while not self.is_action_valid(observation, action):
                    action = np.random.choice(np.arange(self.num_actions))

                next_observation, reward, terminated, truncated, info = self.env.step(action)
                # print(next_observation, observation)
                # print(action)
                # print(self.reward_grid[next_observation, observation, action])
                # print(reward)
                # print("___________________________________")

                eta = 1 / (1 + num_updates[observation, action])
                best_next_action = np.argmax(Q[next_observation])
                Q[observation, action] += eta * (
                        reward + gamma * Q[next_observation, best_next_action] - Q[observation, action])
                num_updates[observation, action] += 1
                observation = next_observation

                if terminated or truncated:
                    break

                if self.after_first:
                    if reward > 0:
                        break

            epsilon *= 0.9999

        optimal_policy = np.argmax(Q, axis=1)
        return optimal_policy, self.starting_index
