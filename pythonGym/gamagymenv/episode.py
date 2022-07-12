

# Class that store the agent's observations, actions and received rewards from a given episode
class Episode:
	def __init__(self):
		self.bounds = None
		self.rewards = None
		self.actions = None
		self.observations = None
		self.last_observation = None
		self.id = None
		self.clear()

	# Set id for the episode
	def set_id(self, episode_id):
		self.id = episode_id

	# Set last observation of the episode
	def set_last_observation(self, last_observation):
		self.last_observation = last_observation

	# Resets/restarts the episode memory buffer
	def clear(self):
		self.observations = []
		self.actions = []
		self.rewards = []
		self.bounds = []
		self.id = -1
		self.last_observation = -1

	# Add observations, actions, rewards to memory
	def add_experience(self, new_observation, new_action, new_reward, bound=None):
		self.observations.append(new_observation)
		self.actions.append(new_action)
		self.rewards.append(new_reward)
		if bound is not None:
			self.bounds.append(bound)

