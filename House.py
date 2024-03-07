from Building import Building
import constants


class House(Building):
    def __init__(self, x, y):
        super().__init__(x, y, 1, 1, constants.WHITE)
