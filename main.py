import random
import sys

import numpy as np
import pygame

import constants
from Building import Building
from House import House
from Townhall import Townhall

from Qlearning import Qlearning

pygame.init()
screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
pygame.display.set_caption("StreetStarter.ai")

screen.fill(constants.BLACK)
clock = pygame.time.Clock()


def check_building_collision(new_building, buildings):
    new_building_rect = new_building.get_rect()
    for building in buildings:
        if new_building_rect.colliderect(building.get_rect()):
            return True
    return False


def check_if_out_of_bounds(new_building):
    x, y = new_building.get_position()
    w, h = new_building.get_size()
    if x + w > constants.GRID_WIDTH or y + h > constants.GRID_HEIGHT:
        return True
    return False


def paint(policy, start_x, start_y):
    # TODO: cut off extra path
    grid = np.array(policy).reshape(constants.GRID_WIDTH, constants.GRID_HEIGHT)

    # Map the values in the grid according to the specified mapping
    mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    direction_array = np.vectorize(mapping.get)(grid)

    current_x, current_y = start_x, start_y
    i = 0
    while i < 70:
        direction = direction_array[current_y, current_x]
        color = constants.BLUE
        rect = pygame.Rect(current_x * constants.GRID_SIZE, current_y * constants.GRID_SIZE,
                           constants.GRID_SIZE, constants.GRID_SIZE)
        pygame.draw.rect(screen, color, rect)

        # Update current position based on direction
        if direction == 'up' and current_y > 0:
            current_y -= 1
        elif direction == 'down' and current_y < direction_array.shape[0] - 1:
            current_y += 1
        elif direction == 'left' and current_x > 0:
            current_x -= 1
        elif direction == 'right' and current_x < direction_array.shape[1] - 1:
            current_x += 1
        i += 1

    # Update the display
    pygame.display.flip()
    clock.tick(60)


def main():
    buildings = []
    rotated = False
    selected_building_type = Building
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    selected_building_type = House
                elif event.key == pygame.K_t:
                    selected_building_type = Townhall
                elif event.key == pygame.K_r:
                    rotated = not rotated
                # begin q-learning with the current input buildings
                elif event.key == pygame.K_RETURN:
                    qlearning = Qlearning(constants.GRID_WIDTH, constants.GRID_HEIGHT, buildings)
                    best_policy, starting_index = qlearning.train(100000)
                    print(best_policy.tolist())
                    # print(Q.tolist())
                    paint(best_policy, starting_index % constants.GRID_WIDTH, starting_index // constants.GRID_WIDTH)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    x //= constants.GRID_SIZE
                    y //= constants.GRID_SIZE
                    new_building = selected_building_type(x, y)
                    if rotated:
                        new_building.rotate()
                    if not check_building_collision(new_building, buildings) and not check_if_out_of_bounds(
                            new_building):
                        buildings.append(new_building)
                    else:
                        continue

        for i in range(constants.GRID_WIDTH + 1):
            pygame.draw.line(screen, constants.WHITE, (i * constants.GRID_SIZE, 0),
                             (i * constants.GRID_SIZE, constants.SCREEN_HEIGHT), 1)
        for j in range(constants.GRID_HEIGHT + 1):
            pygame.draw.line(screen, constants.WHITE, (0, j * constants.GRID_SIZE),
                             (constants.SCREEN_WIDTH, j * constants.GRID_SIZE), 1)

        for building in buildings:
            building.draw(screen)

        pygame.display.flip()


if __name__ == "__main__":
    main()
