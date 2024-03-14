import pygame

import constants


class Building:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (
            self.x * constants.GRID_SIZE, self.y * constants.GRID_SIZE, self.width * constants.GRID_SIZE,
            self.height * constants.GRID_SIZE))

    def rotate(self):
        self.width, self.height = self.height, self.width

    def get_rect(self):
        return pygame.Rect(self.x * constants.GRID_SIZE, self.y * constants.GRID_SIZE, self.width * constants.GRID_SIZE,
                           self.height * constants.GRID_SIZE)
    
    def get_position(self):
        return self.x, self.y
    
    def get_size(self):
        return self.width, self.height
