import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import Sequential
import tensorflow_probability as tfp


class Policy:
    """Class that applies the policy for deep reinforcement learning.

    Longer class information....
    Longer class information....

    Attributes:
        model: the neural network over which we apply the policy
    """
    def __init__(self, model):
        self.model: Sequential = tf.keras.models.clone_model(model)
        self.model.set_weights(model.get_weights())

    # Choose an action given some observations
    # single is used to specify if it's online training
    def choose_action(self, raw_observation: npt.NDArray[np.float64], nb_actions: int, single=True):
        try:
              # add batch dimension to the observation if only a single example was provided
              raw_observation = np.expand_dims(raw_observation, axis=0) if single else raw_observation
              # call to preprocess the raw observation
              observation = self.process_raw_observation(raw_observation)
              # feed the observations through the model to predict the mean and log sigma of each action
              distributions_params = self.model(observation)
              mus, logsigmas = tf.split(distributions_params, 2, axis=1)
              mussigmoid = tf.sigmoid(mus) #conversion
              max_std = 0.3
              min_std = 0.005
              logsigmassigmoid = max_std*tf.sigmoid(logsigmas) + min_std #conversion
              distributions = tfp.distributions.TruncatedNormal(mussigmoid, logsigmassigmoid, low=[0], high=[1])
              actions = distributions.sample()
              return actions.numpy().flatten(), observation.flatten()
        except Exception as e:
              print('Exception handled in choose_action', e)   

    def process_raw_observation(self, raw_observation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Function that process the raw observations obtained from the environment. 
        Default implementation does not process
        Particular implementations to extend the policy method
               :param raw_observation: All observations received from the environment

               :returns: processed observations 
        """
        return raw_observation
