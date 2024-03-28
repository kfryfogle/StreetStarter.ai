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

    def create_rewards(self, buildings):
        print("Enter Key Pressed")
        # max_distance = 4
        max_reward = 200
        # reward_decrement = 2

        # def add_surrounding_cells_to_queue(x, y, distance):
        #     for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #         nx, ny = x + dx, y + dy
        #         if 0 <= nx < constants.GRID_HEIGHT and 0 <= ny < constants.GRID_WIDTH:
        #             if (nx, ny, distance) not in visited:
        #                 queue.append((nx, ny, distance + 1))
        #                 visited.add((nx, ny, distance + 1))
        #
        # for building in buildings:
        #     x, y = building.get_position()
        #     w, h = building.get_size()
        #     queue = deque()
        #     visited = set()
        #
        #     for bx in range(x, x + w):
        #         for by in range(y, y + h):
        #             add_surrounding_cells_to_queue(bx, by, 0)
        #
        #     print(queue)
        #
        #     while queue:
        #         cx, cy, distance = queue.popleft()
        #         if distance > max_distance:
        #             continue
        #         self.reward_grid[cx, cy] = max(0, max_reward - reward_decrement * distance)
        #         add_surrounding_cells_to_queue(cx, cy, distance)

        # for building in buildings:
        #     bx, by = building.get_position()
        #     w, h = building.get_size()
        #     for y in range(max(0, by - max_distance), min(constants.GRID_HEIGHT, by + h + max_distance)):
        #         for x in range(max(0, bx - max_distance), min(constants.GRID_WIDTH, bx + w + max_distance)):
        #
        #             distance = max(abs(x - bx), abs(y - by))
        #             if distance <= max_distance:
        #                 self.reward_grid[y, x] = max(0, max_reward - distance * reward_decrement)

        return self.reward_grid

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
        first_building_width = buildings[1].get_size()[0]
        first_building_height = buildings[1].get_size()[1]
        first_building_x = buildings[1].get_position()[0]
        first_building_y = buildings[1].get_position()[1]

        left_edges = []
        right_edges = []
        top_edges = []
        bottom_edges = []
        # Get left and right edges
        for i in range(first_building_height):
            left_edges.append(first_building_x + grid_width * first_building_y + i * grid_width - 1)
            right_edges.append(first_building_x + grid_width * first_building_y + i * grid_width + first_building_width)

        # Get top and bottom edges
        for i in range(first_building_width):
            top_edges.append(first_building_x + grid_width * first_building_y - grid_width + i)
            bottom_edges.append(
                first_building_x + grid_width * first_building_y + first_building_height * grid_width + i)

        print(left_edges)
        print(right_edges)
        print(top_edges)
        print(bottom_edges)

        for i in range(len(left_edges)):
            if left_edges[i] - 1 >= grid_width * (i + first_building_y):
                self.reward_grid[left_edges[i], left_edges[i] - 1, 3] = 200
            if right_edges[i] + 1 <= grid_width * (i + first_building_y + 1) - 1:
                self.reward_grid[right_edges[i], right_edges[i] + 1, 2] = 200

        for i in range(len(top_edges)):
            if top_edges[i] - grid_width >= 0:
                self.reward_grid[top_edges[i], top_edges[i] - grid_width, 1] = 200
            if bottom_edges[i] + grid_width <= self.num_states - 1:
                self.reward_grid[bottom_edges[i], bottom_edges[i] + grid_width, 0] = 200

        self.reward_grid[top_edges[0], top_edges[0] - 1, 3] = 200
        self.reward_grid[top_edges[-1], top_edges[-1] + 1, 2] = 200
        self.reward_grid[left_edges[0], top_edges[0] - 1, 1] = 200
        self.reward_grid[right_edges[0], top_edges[-1] + 1, 1] = 200
        self.reward_grid[left_edges[-1], bottom_edges[0] - 1, 0] = 200
        self.reward_grid[bottom_edges[0], bottom_edges[0] - 1, 3] = 200
        self.reward_grid[right_edges[-1], bottom_edges[-1] + 1, 0] = 200
        self.reward_grid[bottom_edges[-1], bottom_edges[-1] + 1, 2] = 200
        return self.reward_grid
      
        # for i in range(constants.GRID_HEIGHT):
        #     for j in range(constants.GRID_WIDTH):
        #         self.reward_grid[i][j] = -1
        #
        # first_building_to_connect_width = buildings[1].get_size()[0]
        # first_building_to_connect_height = buildings[1].get_size()[1]
        # first_building_to_connect_x = buildings[1].get_position()[0]
        # first_building_to_connect_y = buildings[1].get_position()[1]
        # print(first_building_to_connect_x, first_building_to_connect_y, first_building_to_connect_width,
        #       first_building_to_connect_height)
        # for i in range(first_building_to_connect_height):
        #     # Check if the left edge of the rectangle is within the bounds of the grid
        #     if first_building_to_connect_x - 1 >= 0:
        #         self.reward_grid[first_building_to_connect_y + i][first_building_to_connect_x - 1] = 200
        #
        #     # Check if the right edge of the rectangle is within the bounds of the grid
        #     if first_building_to_connect_x + first_building_to_connect_width < len(self.reward_grid[0]):
        #         self.reward_grid[first_building_to_connect_y + i][
        #             first_building_to_connect_x + first_building_to_connect_width] = 200
        #
        # for i in range(first_building_to_connect_width):
        #     # Check if the top edge of the rectangle is within the bounds of the grid
        #     if first_building_to_connect_y - 1 >= 0:
        #         self.reward_grid[first_building_to_connect_y - 1][first_building_to_connect_x + i] = 200
        #
        #     # Check if the bottom edge of the rectangle is within the bounds of the grid
        #     if first_building_to_connect_y + first_building_to_connect_height < len(self.reward_grid):
        #         self.reward_grid[first_building_to_connect_y + first_building_to_connect_height][
        #             first_building_to_connect_x + i] = 200
        #
        # return self.reward_grid
    
    def create_reward_after_first(self, paths):
        rewards = paths.copy()
        rewards[rewards == 1] = 200
        rewards[rewards == 0] = -1
        return rewards
        # grid = np.array(policy).reshape(constants.GRID_WIDTH, constants.GRID_HEIGHT)
        # self.reward_grid = np.zeros((self.num_states, self.num_states, self.num_actions))
        # num_actions = 4
        # for i in range(self.num_states):
        #     for j in range(self.num_states):
        #         for action in range(num_actions):
        #             self.reward_grid[i, j, action] = -1

        # # Map the values in the grid according to the specified mapping
        # mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        # direction_array = np.vectorize(grid)

        # current_x, current_y = start_x, start_y
        # i = 0
        # while i < 70:
        #     direction = direction_array[current_y, current_x]
        #     # color = constants.BLUE
        #     # rect = pygame.Rect(current_x * constants.GRID_SIZE, current_y * constants.GRID_SIZE,
        #     #                 constants.GRID_SIZE, constants.GRID_SIZE)
        #     # pygame.draw.rect(screen, color, rect)
        #     self.reward_grid[current_x,current_y, direction] = 200
        #     direction = map.get(direction)

        #     # Update current position based on direction
        #     if direction == 'up' and current_y > 0:
        #         current_y -= 1
        #     elif direction == 'down' and current_y < direction_array.shape[0] - 1:
        #         current_y += 1
        #     elif direction == 'left' and current_x > 0:
        #         current_x -= 1
        #     elif direction == 'right' and current_x < direction_array.shape[1] - 1:
        #         current_x += 1
        #     i += 1        
        # return self.reward_grid


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
