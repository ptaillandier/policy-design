import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import Sequential
from numpy.random import seed


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
    def choose_action(self, raw_observation: npt.NDArray[np.float64], nb_actions: int, single=True)\
            -> (np.float64 | npt.NDArray[np.float64], npt.NDArray[np.float64]):

        tf.random.set_seed(1)
        seed(1)
        # add batch dimension to the observation if only a single example was provided
        raw_observation = np.expand_dims(raw_observation, axis=0) if single else raw_observation
        # call to preprocess the raw observation
        observation = self.process_raw_observation(raw_observation)
        # feed the observations through the model to predict the log probabilities of each possible action
        logits = self.model.predict(observation)
        print('logits', logits)
        # choose an action from the categorical distribution defined by the log probabilities of each possible action
        action = tf.random.categorical(logits, num_samples=nb_actions, seed=1)
        # action = action.numpy().flatten()
        print('action', action)
        if single:
            action = action[0]
        return action, observation.flatten()


    def process_raw_observation(self, raw_observation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Function that process the raw observations obtained from the environment. 
        Default implementation does not process
        Particular implementations to extend the policy method
               :param raw_observation: All observations received from the environment

               :returns: processed observations 
        """
        return raw_observation
