import numpy as np
import pygame

class PyGametoNumpy():
	def __init__():
		pass

	def convert_to_numpy(width, height, buildings):
		print("Enter Key Pressed")
		map = np.zeros((height, width))
		for building in buildings:
			x, y = building.get_position()
			w, h = building.get_size()
			# if building is a townhall, set map value to 2
			if building.color == (255, 0, 0):
				map[y:y+h, x:x+w] = 2
			else:
				map[y:y+h, x:x+w] = 1
		return map
