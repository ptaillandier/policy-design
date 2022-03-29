import numpy as np
import tensorflow as tf 


### Episode ###
# Class that store the agent's observations, actions and received rewards from a given episode
class Episode:

    rewards: []
    actions: []
    observations: []

    def __init__(self):
        self.clear()

    # Resets/restarts the episode memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    R = 0.0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return discounted_rewards


# Helper function that normalizes an np.array x
def normalize(x):
    x = x.astype(np.float32)
    x -= np.mean(x)
    x /= np.std(x)
    return x


### Loss function ###
# Arguments:
#   logits: network's predictions for actions to take 
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits: tf.Tensor, actions, rewards):
    # Note: using softmax activation and SparseCategoricalCrossentropy() has issues and which are patched by the tf.keras model. 
    # A safer approach, in general, is to use a linear output (no activation function) with SparseCategoricalCrossentropy(from_logits=True).
    # Compute the negative log probabilities
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)

    # Scale the negative log probability by the rewards
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

