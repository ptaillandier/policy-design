import tensorflow as tf
import utils
import tensorflow_probability as tfp
import numpy as np
import action_distributions

class ActorTraining:
    """Class that encodes the deep reinforcement learning training logic.

    Longer class information....
    Longer class information....

    Attributes:
        model: the neural network over which we apply the learning
        optimizer: the keras optimizer for the deep learning training
        discount_factor: discount factor hyperparameter for the deep reinforcement learning rewards
    """
    
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(1e-3), clipping_ratio=0.3, target_kl = 0.01):
        """Constructor.

        Args:
           model: model
           optimizer: keras optimizer
           gamma: discounting factor
        """
        self.model = model
        self.optimizer = optimizer
        self.clipping_ratio = clipping_ratio
        self.target_kl = target_kl

    def train(self, episodes, batch_deltas, train_policy_iterations):
        """Training step function (forward and backpropagation).

        Args:
            episodes: set of episodes
        """
        batch_deltas = np.concatenate(batch_deltas)
        advantage_mean = np.mean(batch_deltas)
        advantage_std = np.std(batch_deltas)
        #Normalize advantages
        advantages = tf.truediv(tf.subtract(batch_deltas, advantage_mean), advantage_std)
        actions = []
        discounted_rewards = []
        observations = []

        for episode in episodes:
             actions.append(episode.actions)
             observations.append(episode.observations)

        #Compute joint neglogprobabilities
        joint_neg_logprob = ActorTraining.compute_joint_neglogprob(self.model, np.concatenate(actions), np.vstack(observations))
        print('prob of joint actions', tf.exp(-joint_neg_logprob))
        print('neglogprob of joint actions', joint_neg_logprob)

        #Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = self.train_step(np.vstack(observations), np.concatenate(actions), advantages, joint_neg_logprob)
            if kl > 1.5 * self.target_kl:
                print('Early stopping')
                # Early stopping
                break

    def compute_joint_neglogprob(model, actions, observations):
        distributions_params = model(observations)
        logcon, mus, logsigmas = tf.split(distributions_params, [4, 3, 3], axis=1)
        actions_budget, actions_theta = np.array_split(actions, [4], axis=1)
        print('logcon', logcon)
        #We create the dirichlet distribution with the logcon
        dirichlet_distribution = action_distributions.Dirichlet(logcon)
        print('actions_budget', actions_budget)
        neglogprobbudget = -1*dirichlet_distribution.log_prob(actions_budget)
        print('neglogprobbudget', neglogprobbudget)
        max_std = 0.3
        min_std = 0.005
        logsigmas = tf.sigmoid(logsigmas)*max_std + min_std
        mus = tf.sigmoid(mus)
        print('mus', mus)
        print('logsigmas', logsigmas)
        SMALL_NUMBER = 1e-5
        theta_distributions = action_distributions.SquashedGaussian(mus, logsigmas, low=0.0-SMALL_NUMBER, high=1.0+SMALL_NUMBER)
        print('actions_theta', actions_theta)
        neglogprobthetas = -1*theta_distributions.log_prob(actions_theta)
        print('neglogprobthetas', neglogprobthetas)
        return neglogprobbudget+neglogprobthetas

    #@tf.function 
    def train_step(self, observations, actions, advantage_buffer, neglogprobability_buffer):
        """Training step function (forward and backpropagation).

        Args:
            observations: observations
            actions: actions
            rewards: rewards
        """
        with tf.GradientTape() as tape:
              step_neglogprobability = compute_joint_neglogprob(self.model, actions, observations)
              ratio = tf.exp(-step_neglogprobability + neglogprobability_buffer)
              print('ratio', ratio)
              min_advantage = tf.where(
                      advantage_buffer > 0,
                      (1 + self.clipping_ratio) * advantage_buffer,
                      (1 - self.clipping_ratio) * advantage_buffer,
              )
              print('min advantage', min_advantage)
              print('advantage_buffer', advantage_buffer)
              loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
              
              # Call the compute_loss function to compute the loss
              loss = ActorTraining.compute_loss(logprobability_buffer, actions, advantage_buffer)

        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.model.trainable_variables)
        #print('grads', grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
