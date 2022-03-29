import tensorflow as tf
import utils


class Training:
    """Class that encodes the deep reinforcement learning training logic.

    Longer class information....
    Longer class information....

    Attributes:
        model: the neural network over which we apply the learning
        optimizer: the keras optimizer for the deep learning training
        gamma: discounting factor for the deep reinforcement learning rewards
    """
    
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(1e-3), gamma=0.95):
        """Constructor.

        Args:
           model: model
           optimizer: keras optimizer
           gamma: discounting factor
        """
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        tf.random.set_seed(1)
 
    def train_step(self, observations, actions, rewards):
        """Training step function (forward and backpropagation).

        Args:
            observations: observations
            actions: actions
            rewards: rewards
        """
        with tf.GradientTape() as tape:
            # Compute normalized, discounted, cumulative rewards 
            # What we ultimately care about is which actions are better relative to other actions taken in that episode 
            # so, we'll normalize our computed rewards, using the mean and standard deviation of the rewards across the learning episode.
            discounted_rewards = utils.normalize(utils.discount_rewards(rewards, self.gamma))

            # Forward propagate through the agent network
            logits = self.model(observations)

            # Call the compute_loss function to compute the loss
            loss = utils.compute_loss(logits, actions, discounted_rewards)
      
        # Run backpropagation to minimize the loss using the tape.gradient method
        grads = tape.gradient(loss, self.model.trainable_variables) 
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
