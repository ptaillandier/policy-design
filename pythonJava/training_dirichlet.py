import tensorflow as tf
import utils
import tensorflow_probability as tfp
import numpy as np
import action_distributions

class Training:
    """Class that encodes the deep reinforcement learning training logic.

    Longer class information....
    Longer class information....

    Attributes:
        model: the neural network over which we apply the learning
        optimizer: the keras optimizer for the deep learning training
        discount_factor: discount factor hyperparameter for the deep reinforcement learning rewards
    """
    
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(1e-7), discount_factor=0.95):
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

    #@tf.function 
    def train_step(self, observations, actions, discounted_rewards):
        """Training step function (forward and backpropagation).

        Args:
            observations: observations
            actions: actions
            rewards: rewards
        """
        with tf.GradientTape() as tape:
              # Forward propagate through the agent network
              distributions_params = self.model(observations)
              logcon, mus, logsigmas = tf.split(distributions_params, [3, 3, 3], axis=1)
              actions_budget, actions_theta = np.array_split(actions, [3], axis=1)
              #print('logcon', logcon)
              #We create the dirichlet distribution with the logcon
              dirichlet_distribution = action_distributions.Dirichlet(logcon)
              #print('actions_budget', actions_budget)
              neglogprobbudget = -1*dirichlet_distribution.log_prob(actions_budget)
              #print('neglogprobbudget', neglogprobbudget)
              max_std = 0.3
              min_std = 0.005
              logsigmas = tf.sigmoid(logsigmas)*max_std + min_std
              mus = tf.sigmoid(mus)
              #print('mus', mus)
              #print('logsigmas', logsigmas)
              SMALL_NUMBER = 1e-5
              theta_distributions = action_distributions.SquashedGaussian(mus, logsigmas, low=0.0-SMALL_NUMBER, high=1.0+SMALL_NUMBER)
              #print('actions_theta', actions_theta)
              neglogprobthetas = -1*theta_distributions.log_prob(actions_theta)
              #print('neglogprobthetas', neglogprobthetas)
              #distributions = tfp.distributions.TruncatedNormal(tf.multiply(tf.sigmoid(mus), bounds), tf.sigmoid(logsigmas)*max_std + min_std, low=[0], high=bounds)
              # Call the compute_loss function to compute the loss
              loss = Training.compute_loss(neglogprobbudget+neglogprobthetas, actions, discounted_rewards)

        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.model.trainable_variables)
        #print('grads', grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    

    ### Policy gradient loss function ###
    # Arguments:
    #   distributions: network's probability distributions for actions to take 
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    #@staticmethod
    #@tf.function
    def compute_loss(neglogprob, actions, rewards):
        #print('prob of joint actions', tf.exp(-neglogprob))
        #print('neglogprob of joint actions', neglogprob)
        #print('rewards', rewards)
        # Compute the negative log probabilities
        #neg_logprob = -1*distributions.log_prob(actions)
        #neg_logprob = tf.reduce_sum(neg_logprob, axis=1)
        # Scale the negative log probability by the rewards
        #loss = tf.reduce_mean(tf.cast(neg_logprob, tf.float64) * rewards)
        loss = tf.reduce_mean(neglogprob * rewards)
        return loss

