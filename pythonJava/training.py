import tensorflow as tf
import utils
import tensorflow_probability as tfp
import numpy as np

class Training:
    """Class that encodes the deep reinforcement learning training logic.

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

    def train(self, episodes):
        """Training step function (forward and backpropagation).

        Args:
            episodes: set of episodes
        """
        actions = []
        discounted_rewards = []
        observations = []

        for episode in episodes:
             actions.append(episode.actions)
             # Compute normalized, discounted, cumulative rewards 
             # What we ultimately care about is which actions are better relative to other actions taken in that episode 
             #-- so, we'll normalize our computed rewards, using the mean and standard deviation of the rewards across the learning episode.
             discounted_rewards.append(utils.discount_rewards(episode.rewards, self.discount_factor))
             observations.append(episode.observations)
        #When normalizing I think we normalize through the whole set of discounted rewards        
        self.train_step(np.vstack(observations), np.concatenate(actions), utils.normalize(np.concatenate(discounted_rewards)))

    @tf.function 
    def train_step(self, observations, actions, discounted_rewards):
        """Training step function (forward and backpropagation).

        Args:
            observations: observations
            actions: actions
            rewards: rewards
        """
        with tf.GradientTape() as tape:
              #print('self.model.get_weights()', self.model.get_weights())
              # Forward propagate through the agent network
              distributions_params = self.model(observations)
              mus, logsigmas = tf.split(distributions_params,2, axis=1)
              max_std = 0.3
              min_std = 0.005
              distributions = tfp.distributions.TruncatedNormal(tf.sigmoid(mus), tf.sigmoid(logsigmas)*max_std + min_std, low=[0], high=[1])
              # Call the compute_loss function to compute the loss
              loss = Training.compute_loss(distributions, actions, discounted_rewards)

        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    

    ### Policy gradient loss function ###
    # Arguments:
    #   distributions: network's probability distributions for actions to take 
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    #@staticmethod
    @tf.function
    def compute_loss(distributions, actions, rewards):
         # Compute the negative log probabilities
        neg_logprob = -1*distributions.log_prob(actions)
        neg_logprob = tf.reduce_sum(neg_logprob, axis=1)
        # Scale the negative log probability by the rewards
        #loss = tf.reduce_mean(tf.cast(neg_logprob, tf.float64) * rewards)
        loss = tf.reduce_mean(neg_logprob * rewards)
        return loss

