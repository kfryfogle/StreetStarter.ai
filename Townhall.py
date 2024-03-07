import constants
from Building import Building


class Townhall(Building):
    def __init__(self, x, y):
        super().__init__(x, y, 3, 5, constants.RED)
