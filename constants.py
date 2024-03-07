import pygame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Grid
GRID_SIZE = 30
GRID_WIDTH = 40
GRID_HEIGHT = 40

# Game
BUILDINGS = {1: "House", 2: "Townhall"}
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE

# Q-learning
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPISODES = 1000
STEPS_PER_EPISODE = 100
