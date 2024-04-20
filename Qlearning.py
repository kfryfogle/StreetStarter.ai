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

        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
        if after_first:
            self.reward_grid = gtnp.create_reward_after_first(paths)
            self.create_transition_matrix(paths)
        else:
            self.reward_grid, first_building_edges, left_edges, bottom_edges, top_edges, right_edges = gtnp.create_reward_first_step(
                buildings)
            self.create_transition_matrix_first_step(first_building_edges, left_edges, bottom_edges, top_edges,
                                                     right_edges)

        # Create P_0 for starting state distribution
        self.P_0 = np.array([0 for _ in range(self.num_states)])
        self.starting_index = self.find_starting_index()
        self.P_0[self.starting_index] = 1
        self.env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, r=self.reward_grid, p=self.T)

    def find_starting_index(self):
        # Find distance to right edge
        dist_to_right = self.width - (self.current_building.get_position()[0] + self.current_building.get_size()[0])

        # Find distance to left edge
        dist_to_left = self.current_building.get_position()[0]

        # Find distance to top edge
        dist_to_top = self.current_building.get_position()[1]

        # Find distance to bottom
        dist_to_bottom = self.height - (self.current_building.get_position()[1] + self.current_building.get_size()[1])

        max_dist = max(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        print(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        print(max_dist)
        if max_dist == dist_to_left:
            starting_index = self.current_building.get_position()[0] - 1 + self.current_building.get_position()[
                1] * self.width
        elif max_dist == dist_to_right:
            starting_index = self.current_building.get_position()[0] + self.current_building.get_size()[0] + \
                             self.current_building.get_position()[1] * self.width
        elif max_dist == dist_to_top:
            starting_index = self.current_building.get_position()[0] + (
                        self.current_building.get_position()[1] - 1) * self.width
        else:
            starting_index = self.current_building.get_position()[0] + (
                        self.current_building.get_position()[1] + self.current_building.get_size()[1]) * self.width
        return starting_index

    def all_neighbours(self, state):
        neighbours = {}
        if state - self.width >= 0:
            neighbours[0] = state - self.width
        if state + self.width <= self.num_states - 1:
            neighbours[1] = state + self.width
        if state - 1 >= 0:
            neighbours[2] = state - 1
        if state + 1 <= self.num_states - 1:
            neighbours[3] = state + 1
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

    def is_adjacent_to_building(self, i, j):
        if (i > 0 and self.map[i - 1, j] != 0) or (i < self.width - 1 and self.map[i + 1, j] != 0) or (
                j > 0 and self.map[i, j - 1] != 0) or (j < self.height - 1 and self.map[i, j + 1] != 0):
            return True
        else:
            return False

    def create_transition_matrix(self, paths):
        for x in range(self.width):
            for y in range(self.height):
                neighbors = self.valid_neighbours(x, y)
                for action in range(self.num_actions):
                    if action in neighbors:
                        self.T[neighbors[action][0] * self.width + neighbors[action][
                            1], x * self.width + y, action] = 1

        indices = np.where(paths == 1)
        x_coord = indices[0]
        y_coord = indices[1]
        for i in range(len(x_coord)):
            state = x_coord[i] + (constants.GRID_WIDTH * y_coord[i])
            self.T[:, state, :] = 0

    def create_transition_matrix_first_step(self, first_building_edges, left_edges, bottom_edges, top_edges,
                                            right_edges):
        for x in range(self.width):
            for y in range(self.height):
                state = x * self.width + y
                neighbors = self.all_neighbours(state)
                if state in first_building_edges:
                    continue
                for action in range(self.num_actions):
                    if action in neighbors:
                        self.T[neighbors[action], state, action] = 1

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
        neighbors = self.valid_neighbours(x, y)
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
        def action_hits_wall(state_from, action):
            movements = {
                0: (0, -1),  # Move up
                1: (0, 1),  # Move down
                2: (-1, 0),  # Move left
                3: (1, 0)  # Move right
            }
            row, col = state_from % self.width, state_from // self.width
            movement = movements[action]
            next_row = row + movement[0]
            next_col = col + movement[1]
            if next_row < 0 or next_row >= self.height or next_col < 0 or next_col >= self.width:
                return True
            return False

        Q = np.random.uniform(low=-0.005, high=0.005, size=(self.num_states, self.num_actions))
        num_updates = np.zeros((self.num_states, self.num_actions))

        gamma = 0.97
        epsilon = 0.97

        observation, info = self.env.reset()
        ending_points = set()
        optimal_policy = np.full(self.num_states, -1)

        for state_from in range(self.num_states):
            for action in range(self.num_actions):
                if action_hits_wall(state_from, action):
                    Q[state_from, action] = -5000

        for i in tqdm(range(num_episodes)):
            observation, info = self.env.reset()
            visited = set()
            total_reward = 0
            while True:
                explr_explo = random.choices([1, 0], weights=[epsilon, 1 - epsilon], k=1)
                if explr_explo[0] == 1:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    action = np.argmax(Q[observation])

                considered_actions = set()
                while not self.is_action_valid(observation, action) and len(considered_actions) < 4:
                    action = np.random.choice(np.arange(self.num_actions))
                    considered_actions.add(action)

                if len(considered_actions) == 4:
                    break

                if not action_hits_wall(observation, action):
                    next_observation, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward

                    visited.add(observation)

                    eta = 1 / (1 + num_updates[observation, action])
                    best_next_action = np.argmax(Q[next_observation])

                    Q[observation, action] = ((1 - eta) * (Q[observation, action]) +
                                              eta * (reward + gamma * Q[next_observation, best_next_action]))

                    num_updates[observation, action] += 1
                    observation = next_observation

                    if reward > 0:
                        ending_points.add((observation, total_reward))
                        break

            epsilon *= 0.9999

        optimal_policy = np.argmax(Q, axis=1)
        final_ending_point = max(ending_points, key=lambda x: x[1])[0]
        print(ending_points)
        return optimal_policy, self.starting_index, final_ending_point

    def move_near_visited_building(self, next_observation, visited_buildings):
        x, y = self.current_coordinates(next_observation)
        neighbors = self.all_neighbours(x, y)
        visited_coordinates = []

        new_X = next_observation // self.height
        new_Y = next_observation % self.width

        # print("new coordinates: ", new_X, new_Y)
        # get all coordinates of the current visited buildings
        for building in visited_buildings:
            x = building.get_position()[0]
            y = building.get_position()[1]

            if new_X == x + 1 or new_X == x - 1 or new_Y == y + 1 or new_Y == y - 1:
                return True

        return False
