#From: https://github.com/ray-project/ray/blob/master/rllib/models/tf/tf_action_dist.py
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from math import log
# Min and Max outputs (clipped) from an NN-output layer interpreted as the
# log(x) of some x (e.g. a stddev of a normal
# distribution).
MIN_LOG_NN_OUTPUT = -5
MAX_LOG_NN_OUTPUT = 2
"""A tanh-squashed Gaussian distribution defined by: mean, std, low, high.
The distribution will never return low or high exactly, but
`low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
"""
SMALL_NUMBER = 1e-6

class SquashedGaussian():
    # Min and Max outputs (clipped) from an NN-output layer interpreted as the
    # log(x) of some x (e.g. a stddev of a normal
    # distribution).
    MIN_LOG_NN_OUTPUT = -5
    MAX_LOG_NN_OUTPUT = 2
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.
    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """
    SMALL_NUMBER = 1e-8

    def __init__(self, mean, std, low: float = -1.0, high: float = 1.0):
        """Parameterizes the distribution via `inputs`.
        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """
        #self.mean, log_std = tf.split(inputs, 2, axis=-1)
        self.mean = mean
        # Clip `scale` values (coming from NN) to reasonable values.
        #log_std = tf.clip_by_value(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        #std = tf.exp(std)
        #print('creating squashed dist with  mean', mean, 'std', std, 'low', low,'high', high)
        self.distr = tfp.distributions.Normal(loc=self.mean, scale=std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high

    def prob(self, x):
        return tf.exp(self.log_prob(x))
     
    def log_prob(self, x):
        # Unsquash values (from [low,high] to ]-inf,inf[)
        unsquashed_values = tf.cast(self._unsquash(x), self.mean.dtype)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.distr.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = tf.clip_by_value(log_prob_gaussian, -100, 100)
        log_prob_gaussian = tf.reduce_sum(log_prob_gaussian, axis=-1)
        # Get log-prob for squashed Gaussian.
        unsquashed_values_tanhd = tf.math.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - tf.reduce_sum(
            tf.math.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), axis=-1
        )
        return log_prob

    def sample(self):
        return self._squash(self.distr.sample())

    def _squash(self, raw_values):
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = ((tf.math.tanh(raw_values) + 1.0) / 2.0) * (
            self.high - self.low
        ) + self.low
        return tf.clip_by_value(squashed, self.low, self.high)
    
    def _unsquash(self, values):
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        # Stabilize input to atanh.
        save_normed_values = tf.clip_by_value(
            normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER
        )
        unsquashed = tf.math.atanh(save_normed_values)
        return unsquashed

class DiagGaussian():
    """Action distribution where each vector element is a gaussian.
    The first half of the input vector defines the gaussian means, and the
    second half the gaussian standard deviations.
    """

    def __init__( self, mean, std ):
        self.mean = tf.convert_to_tensor(mean)
        self.std = tf.convert_to_tensor(std)
        self.log_std = tf.math.log(self.std)
        

    """The log-likelihood of the action x in the distribution."""
    def log_prob(self, x):
        return (
            -0.5
            * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std), axis=1
            )
            - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32)
            - tf.reduce_sum(self.log_std, axis=1)
        )

    def prob(self, x):
        return tf.exp(self.log_prob(x))

    """Draw a sample from the action distribution.""" 
    def sample(self):
        sample = self.mean + self.std * tf.random.normal(tf.shape(self.mean))
        return sample


class Dirichlet():
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.
    e.g. actions that represent resource allocation."""
    SMALL_NUMBER = 1e-6
    def __init__(self, inputs):

        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = 1e-7
        #Some components of the samples can be zero due to finite precision. This happens more often when some of the concentrations are very small. Make sure to round the samples to np.finfo(dtype).tiny before computing the density.
        inputs = tf.clip_by_value(inputs, log(SMALL_NUMBER), -log(SMALL_NUMBER))
        self.concentration = tf.exp(inputs) + self.epsilon
        self.dist = tfp.distributions.Dirichlet(concentration=self.concentration, validate_args=True, allow_nan_stats=False)  

    def log_prob(self, x):
        # Support of Dirichlet are positive real numbers. x is already
        # an array of positive numbers, but we clip to avoid zeros due to
        # numerical errors.
        x = tf.maximum(x, self.epsilon)
        x = x / tf.reduce_sum(x, axis=-1, keepdims=True)
        return self.dist.log_prob(x) 

    def sample(self):
        return self.dist.sample()

    def prob(self, x):
        return tf.exp(self.log_prob(x))

    def get_concentration(self):
        return self.concentration.numpy()
