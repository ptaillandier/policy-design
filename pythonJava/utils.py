import numpy as np
import tensorflow as tf 
import scipy.signal

### Episode ###
# Class that store the agent's observations, actions and received rewards from a given episode
class Episode:
    def __init__(self):
        self.clear()

    # Resets/restarts the episode memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_experience(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


# Compute discounted, cumulative rewards per time step (e.g. the rewards-to-go)
# discounted_rewards[t] = \sum^T_{t' = t} \gamma^{t'-t}*r_t
# Arguments:
#   rewards: reward at timesteps in episode
#   discount_factor
# Returns:
#   discounted reward at timesteps (rewards-to-go) in episode
#   Example:
#   input: 
#        vector x, 
#        [x0, 
#         x1, 
#         x2]
#    output:
#        [x0 + discount * x1 + discount^2 * x2,  
#         x1 + discount * x2,
#         x2]
def discount_rewards(rewards, discount_factor):
    #Replaced previous implementation by rllab more efficient and criptic! one
    return scipy.signal.lfilter([1], [1, float(-discount_factor)], rewards[::-1], axis=0)[::-1]


# Helper function that normalizes an np.array x
def normalize(x):
    x = x.astype(np.float32)
    x -= np.mean(x)
    x /= np.std(x)
    return x



