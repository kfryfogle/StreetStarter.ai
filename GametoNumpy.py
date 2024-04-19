import queue
from collections import deque

import numpy as np
import pygame

import constants


class PyGametoNumpy:
    def __init__(self, width, height, buildings):
        self.map = np.zeros((width, height))
        self.reward_grid = np.zeros((width, height))
        self.buildings = buildings
        self.num_states = width * height
        self.num_actions = 4

    def create_reward_first_step(self, buildings):
        num_actions = 4  # Up, Down, Left, Right
        grid_height = constants.GRID_HEIGHT
        grid_width = constants.GRID_WIDTH

        # Initialize the 3D reward matrix with default reward values
        self.reward_grid = np.zeros((self.num_states, self.num_states, self.num_actions))

        # Set default rewards for each action in each grid cell
        for i in range(self.num_states):
            for j in range(self.num_states):
                for action in range(num_actions):
                    self.reward_grid[i, j, action] = -1  # Default reward

        # Extract building information
        zeroth_building_width = buildings[0].get_size()[0]
        zeroth_building_height = buildings[0].get_size()[1]
        zeroth_building_x = buildings[0].get_position()[0]
        zeroth_building_y = buildings[0].get_position()[1]
        first_building_width = buildings[1].get_size()[0]
        first_building_height = buildings[1].get_size()[1]
        first_building_x = buildings[1].get_position()[0]
        first_building_y = buildings[1].get_position()[1]

        left_edges = []
        right_edges = []
        top_edges = []
        bottom_edges = []
        zeroth_building_left_edges = []
        zeroth_building_right_edges = []
        zeroth_building_top_edges = []
        zeroth_building_bottom_edges = []
        # Get left and right edges
        for i in range(first_building_height):
            left_edges.append(first_building_x + grid_width * first_building_y + i * grid_width - 1)
            right_edges.append(first_building_x + grid_width * first_building_y + i * grid_width + first_building_width)

        # Get top and bottom edges
        for i in range(first_building_width):
            top_edges.append(first_building_x + grid_width * first_building_y - grid_width + i)
            bottom_edges.append(
                first_building_x + grid_width * first_building_y + first_building_height * grid_width + i)

        for i in range(len(left_edges)):
            if left_edges[i] - 1 >= grid_width * (i + first_building_y):
                self.reward_grid[left_edges[i], left_edges[i] - 1, 3] = 2000
            if right_edges[i] + 1 <= grid_width * (i + first_building_y + 1) - 1:
                self.reward_grid[right_edges[i], right_edges[i] + 1, 2] = 2000

        for i in range(len(top_edges)):
            if top_edges[i] - grid_width >= 0:
                self.reward_grid[top_edges[i], top_edges[i] - grid_width, 1] = 2000
            if bottom_edges[i] + grid_width <= self.num_states - 1:
                self.reward_grid[bottom_edges[i], bottom_edges[i] + grid_width, 0] = 2000

        # TODO: fix when one of these indices are out of bounds eg. when building is along an edge of grid
        self.reward_grid[top_edges[0], top_edges[0] - 1, 3] = 2000
        self.reward_grid[top_edges[-1], top_edges[-1] + 1, 2] = 2000
        self.reward_grid[left_edges[0], top_edges[0] - 1, 1] = 2000
        self.reward_grid[right_edges[0], top_edges[-1] + 1, 1] = 2000
        self.reward_grid[left_edges[-1], bottom_edges[0] - 1, 0] = 2000
        self.reward_grid[bottom_edges[0], bottom_edges[0] - 1, 3] = 2000
        self.reward_grid[right_edges[-1], bottom_edges[-1] + 1, 0] = 2000
        self.reward_grid[bottom_edges[-1], bottom_edges[-1] + 1, 2] = 2000

        first_building_edges = left_edges + right_edges + top_edges + bottom_edges

        # Get left and right edges
        for i in range(zeroth_building_height):
            zeroth_building_left_edges.append(zeroth_building_x + grid_width * zeroth_building_y + i * grid_width - 1)
            zeroth_building_right_edges.append(
                zeroth_building_x + grid_width * zeroth_building_y + i * grid_width + zeroth_building_width)

        # Get top and bottom edges
        for i in range(zeroth_building_width):
            zeroth_building_top_edges.append(zeroth_building_x + grid_width * zeroth_building_y - grid_width + i)
            zeroth_building_bottom_edges.append(
                zeroth_building_x + grid_width * zeroth_building_y + zeroth_building_height * grid_width + i)

        for i in range(len(zeroth_building_left_edges)):
            if zeroth_building_left_edges[i] - 1 >= grid_width * (i + zeroth_building_y):
                self.reward_grid[zeroth_building_left_edges[i] + 1, zeroth_building_left_edges[i], 3] = -2000
            if zeroth_building_right_edges[i] + 1 <= grid_width * (i + zeroth_building_y + 1) - 1:
                self.reward_grid[zeroth_building_right_edges[i] - 1, zeroth_building_right_edges[i], 2] = -2000

        for i in range(len(zeroth_building_top_edges)):
            if zeroth_building_top_edges[i] - grid_width >= 0:
                self.reward_grid[zeroth_building_top_edges[i] + grid_width, zeroth_building_top_edges[i], 1] = -2000
            if zeroth_building_bottom_edges[i] + grid_width <= self.num_states - 1:
                self.reward_grid[
                    zeroth_building_bottom_edges[i] - grid_width, zeroth_building_bottom_edges[i], 0] = -2000

        return (self.reward_grid, first_building_edges, zeroth_building_left_edges,
                zeroth_building_bottom_edges, zeroth_building_top_edges, zeroth_building_right_edges)

    def create_reward_after_first(self, paths):
        self.reward_grid = np.full((self.num_states, self.num_states, self.num_actions), -1)
        indices = np.where(paths == 1)
        x_coord = indices[0]
        y_coord = indices[1]
        for i in range(len(x_coord)):
            state = x_coord[i] + constants.GRID_WIDTH * y_coord[i]
            self.reward_grid[state, :, :] = 200
        return self.reward_grid

    def convert_to_numpy(self):
        print("Enter Key Pressed")
        for building in self.buildings:
            x, y = building.get_position()
            w, h = building.get_size()
            # if building is a townhall, set map value to 1
            if building.color == (255, 0, 0):
                self.map[y:y + h, x:x + w] = 1
            else:
                self.map[y:y + h, x:x + w] = 2

        return self.map
