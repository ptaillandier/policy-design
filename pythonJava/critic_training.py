import tensorflow as tf
import utils
import numpy as np
from numpy.random import seed
class CriticTraining:
    """Class that encodes the deep reinforcement learning training logic.
    with the ppo gradient algorithm
    Longer class information....
	    Longer class information....

    Attributes:
        model: the neural network over which we apply the learning
        optimizer: the keras optimizer for the deep learning training
        discount_factor: discount factor hyperparameter for the deep reinforcement learning rewards
    """
    
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(1e-3), discount_factor=0.95):
        """Constructor.

        Args:
           model: model
           optimizer: keras optimizer
           gamma: discounting factor
        """
        self.model = model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
      
    def train(self, episodes, train_value_iterations):
        """Training step function (forward and backpropagation).

        Args:
            episodes: set of episodes
        """
        discounted_rewards = []
        observations = []

        for episode in episodes:
             discounted_rewards.append(utils.discount_rewards(episode.rewards, self.discount_factor))
             observations.append(episode.observations)
        
        observations = np.vstack(observations)
        discounted_rewards = np.concatenate(discounted_rewards) #TODO: verify in the frameworks if there is normalization for the rewards
        
        for _ in range(train_value_iterations):
            self.train_step(observations, discounted_rewards)

    def train_step(self, observations, discounted_rewards):
  
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
               loss = tf.reduce_mean((discounted_rewards - self.model(observations)) ** 2)
        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    

      
