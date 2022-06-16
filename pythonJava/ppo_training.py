import tensorflow as tf
import utils
import tensorflow_probability as tfp
import numpy as np
import action_distributions

class PPOTraining:
    """Class that encodes the deep reinforcement learning training logic.

    Longer class information....
    Longer class information....

    Attributes:
        model: the neural network over which we apply the learning
        optimizer: the keras optimizer for the deep learning training
        discount_factor: discount factor hyperparameter for the deep reinforcement learning rewards
    """
    
    def __init__(self, actor_model, critic_model, actor_optimizer=tf.keras.optimizers.Adam(1e-3), critic_optimizer=tf.keras.optimizers.Adam(1e-3), clipping_ratio=0.3, target_kl = 0.01, discount_factor=0.99, gae_lambda=0.95, minibatch_splitting_method="SHUFFLE_TRANSITIONS"):
        """Constructor.

        Args:
           model: model
           optimizer: keras optimizer
           gamma: discounting factor
        """
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clipping_ratio = clipping_ratio
        self.target_kl = target_kl
        self.discount_factor = discount_factor
        self.minibatch_splitting_method = minibatch_splitting_method
        self.gae_lambda = gae_lambda
        print('self.minibatch_splitting_method ', self.minibatch_splitting_method)

    def compute_advantages_and_returns(self, episode):
         # Compute the values for observations present in the episode
         episode_values = np.squeeze(self.critic_model(np.vstack(episode.observations)))
         # Compute episode advantages
         episode_advantages = utils.gae(episode.rewards, np.append(episode_values,0), self.discount_factor, self.gae_lambda)
         return episode_advantages, (episode_advantages-episode_values)

    def train(self, episodes, n_update_epochs, n_mini_batches):
        """Training step function (forward and backpropagation).

        Args:
            episodes: set of episodes
        """
        actions = []
        discounted_rewards = []
        observations = []
        returns = []
        advantages = []
        for episode in episodes:
            # Compute the values for observations present in the episode
            episode_advantages, episode_returns = self.compute_advantages_and_returns(episode)
            # Add returns
            returns.append(episode_returns)
            # Add advantages
            advantages.append(episode_advantages)            
            # Compute disconted rewards-to-go (by now not used)
            discounted_rewards.append(utils.discount_rewards(episode.rewards, self.discount_factor))
            actions.append(episode.actions)
            observations.append(episode.observations)

        observations = np.vstack(observations)
        actions = np.concatenate(actions)
        returns = np.concatenate(returns)
        advantages = np.concatenate(advantages)
        discounted_rewards = np.concatenate(discounted_rewards)
        #Compute joint neglogprobabilities
        joint_neg_logprob = PPOTraining.compute_joint_neglogprob(self.actor_model, actions, observations)
        #print('prob of joint actions', tf.exp(-joint_neg_logprob))
        #print('neglogprob of joint actions', joint_neg_logprob)
        # Compute index of each element of each mini_batch
        # Create the indices array
        batch_size = len(advantages)
        #print('batch_size ', batch_size)
        assert batch_size % n_mini_batches == 0
        inds = np.arange(batch_size)
        mini_batch_size = batch_size // n_mini_batches
        #print('mini_batch_size ', mini_batch_size)

        #Update the policy and implement early stopping using KL divergence
        for tpi in range(n_update_epochs):
            # Randomize the indexes of the mini_batch for this epoch if SHUFFLE_TRANSITIONS
            if self.minibatch_splitting_method == "SHUFFLE_TRANSITIONS":
                print("SHUFFLing TRANSITIONS")
                np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, batch_size, mini_batch_size):
                 end = start + mini_batch_size
                 mbinds = inds[start:end]
                 print('start ', start, ' end ', end, ' mbinds ', mbinds)
                 #slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                 minibatch_observations = observations[mbinds]
                 minibatch_actions = actions[mbinds]
                 minibatch_advantages = advantages[mbinds]
                 #Normalize advantages at the level of mini_batch
                 mbadvantage_mean = np.mean(minibatch_advantages)
                 mbadvantage_std = np.std(minibatch_advantages) + 1e-8
                 minibatch_advantages = tf.truediv(tf.subtract(minibatch_advantages, mbadvantage_mean), mbadvantage_std)
                 minibatch_joint_neg_logprob = tf.gather(joint_neg_logprob, mbinds)
                 kl = self.actor_train_step( minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_joint_neg_logprob)
                 minibatch_returns = returns[mbinds]
                 self.critic_train_step( minibatch_observations, minibatch_returns)
                 #Recompute advantages and returns 
                 print("RECOMPUTING ADVANTAGES AND RETURNS AT MINIBATCH ITERATION LEVEL")
                 returns = []
                 advantages = []
                 for episode in episodes:
                     # Compute the values for observations present in the episode
                     episode_advantages, episode_returns = self.compute_advantages_and_returns(episode)
                     # Add returns
                     returns.append(episode_returns)
                     # Add advantages
                     advantages.append(episode_advantages)
                 returns = np.concatenate(returns)
                 advantages = np.concatenate(advantages)

                 print('update epoch: ', tpi,'minibatch ', start, ' kl:', kl)
                 if kl > 1.5 * self.target_kl:
                     print('Early stopping at epoch ', tpi)
                     # Early stopping
                     break

    def compute_joint_neglogprob(model, actions, observations):
        distributions_params = model(observations)
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
        return neglogprobbudget+neglogprobthetas


    def critic_train_step(self, observations, returns):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
               loss = tf.reduce_mean((returns - self.critic_model(observations)) ** 2)
        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

    #@tf.function 
    def actor_train_step(self, observations, actions, advantage_buffer, neglogprobability_buffer):
        """Training step function (forward and backpropagation).

        Args:
            observations: observations
            actions: actions
            rewards: rewards
        """
        with tf.GradientTape() as tape:
              step_neglogprobability = PPOTraining.compute_joint_neglogprob(self.actor_model, actions, observations)
              ratio = tf.exp(-step_neglogprobability + neglogprobability_buffer)
              #print('ratio', ratio)
              #min_advantage = tf.where(
              #        advantage_buffer > 0,
              #        (1 + self.clipping_ratio) * advantage_buffer,
              #        (1 - self.clipping_ratio) * advantage_buffer,
              #)
              #print('min advantage', min_advantage)
              #print('advantage_buffer', advantage_buffer)
              p_loss1 = ratio * advantage_buffer
              #print('p_loss1= ratio * advantage_buffer', p_loss1)
              p_loss2 = tf.clip_by_value(ratio, 1 - self.clipping_ratio, 1 + self.clipping_ratio)*advantage_buffer
              #print('p_loss2= clipbyvalue * advantage_buffer', p_loss2)
              loss = -tf.reduce_mean(tf.minimum(p_loss1, p_loss2))
 

        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.actor_model.trainable_variables)
        #print('grads', grads)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))
        kl = tf.reduce_mean(
            -neglogprobability_buffer
            + PPOTraining.compute_joint_neglogprob(self.actor_model, actions, observations)
        )
        kl = tf.reduce_sum(kl)
        return kl #Return KL divergence

