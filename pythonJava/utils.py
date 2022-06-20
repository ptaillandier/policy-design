import numpy as np
import tensorflow as tf 
import scipy.signal
import matplotlib.pyplot as plt

### Episode ###
# Class that store the agent's observations, actions and received rewards from a given episode
class Episode:
    def __init__(self):
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


# Compute GAE advantages
def gae(rewards, values, discount_factor, gae_lambda):
    deltas = rewards + discount_factor*values[1:] - values[:-1]
    advantages = discount_rewards(deltas, discount_factor * gae_lambda)
    return advantages.astype('float32')

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

# Function that builds a customized multilayer perceptron (MLP) neural network
def mlp(n_observations, sizes, activation='relu', output_activation=None, last_layer_scaling=np.sqrt(2)):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(n_observations,)))
    for size in sizes[:-1]:
        x = tf.keras.layers.Dense(units=size, activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)))
        model.add(x)
    model.add(tf.keras.layers.Dense(units=sizes[-1], activation=output_activation, kernel_initializer=tf.keras.initializers.Orthogonal(gain=last_layer_scaling)))
    return model

# Prints full summary of a mlp model, including activation function
def full_summary(layer):
    print('summary for ' + layer.name)
    #check if this layer has activation function
    if hasattr(layer, 'activation'):
        print('layer.activation ', layer.activation)
    #check if this layer has layers
    if hasattr(layer, 'layers'):
        layer.summary()
        for l in layer.layers:
            full_summary(l)

#Plot distribution
def plot_distribution(probs, filepath):
      xstep=1.0/(len(probs)-1)
      x = np.arange(0.0, 1.0+xstep, xstep)
      #Plot the results
      plt.plot(x, probs)
      plt.xlabel('x')
      plt.ylabel('Prob(x)')
      plt.savefig(filepath)
      plt.clf()

#Save plot distribution
def save_plot_distribution(distribution, filepath):
      xstep = 0.01
      x = np.arange(0.0, 1.0+xstep, xstep)
      log_probs = distribution.log_prob(x) 
      probs = np.exp(log_probs)
      plot_distribution(probs, filepath)
