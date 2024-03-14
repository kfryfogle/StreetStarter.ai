import numpy as np
import gymnasium as gym
import matrix_mdp

from GametoNumpy import PyGametoNumpy as gtnp


class Qlearning():
	def __init__(self, width, height, buildings):
		# Hyperparameters
		self.epsilon = 0.9
		self.gamma = 0.9
		self.episodes = 1000
		self.steps_per_episode = 100

		# Mapping of the game
		self.width = width
		self.height = height
		self.buildings = buildings
		self.map = gtnp.convert_to_numpy(self.width, self.height, self.buildings)

		# Create P_0 for starting state distribution
		self.P_0 = np.zeros([self.width * self.height])
		# TODO: determine starting state
		# TODO: determine reward function
		# initialize the Q-table values to 0
		self.q_table = np.zeros([self.width * self.height, 4])
		# initialize number of updates for each state-action pair to 0
		self.num_updates = np.zeros([self.width * self.height, 4])
		self.env = gym.make('matrix_mdp-v0', p_0=self.P_0, r=self.reward, P=self.P)
		self.observation, self.info = self.env.reset()

	
	def train(self):
		pass

		

	