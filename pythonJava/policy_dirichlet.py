import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import Sequential
import tensorflow_probability as tfp
import utils
import action_distributions

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

    #Bound distributions and sample actions respecting budget
    def bound_and_sample_actions(self, logcon, mussigmoid, logsigmassigmoid, budget):
        actions = np.full(6,-1.0, dtype='float32')
        actions_env = np.full(5,-1.0, dtype='float32')
        #We create the dirichlet distribution with the logcon
        dirichlet_distribution = action_distributions.Dirichlet(logcon)
        budget_dist = dirichlet_distribution.sample()
        #print('budget_dist', budget_dist, ' logp ', dirichlet_distribution.log_prob(budget_dist) , 'prob', dirichlet_distribution.prob(budget_dist))
        budget_dist = budget_dist.numpy().flatten()
        actions[0:3] = budget_dist
        SMALL_NUMBER = 1e-5
        #Sample thetas for each axis
        theta_distribution = action_distributions.SquashedGaussian(mussigmoid, logsigmassigmoid, low=0.0-SMALL_NUMBER, high=1.0+SMALL_NUMBER)
        thetas = theta_distribution.sample()
        #print('thetas', thetas, 'logp', theta_distribution.log_prob(thetas), 'prob', theta_distribution.prob(thetas))
        thetas = thetas.numpy().flatten()
        actions[3:6] = thetas[0:3]
        actions_env[0:2] = thetas[0:2]
        actions_env[3] = thetas[2]
        #Compute nmanagement and nenvironment from thetas and budget distribution
        fman = (budget*budget_dist[1])/(thetas[1]*100.0)
        #print('fman', fman)
        fman = np.floor(fman*100)/100
        actions_env[2] = fman
        #print('fmanbounded', fman)
        fenv = (budget*budget_dist[2])/(thetas[2]*100.0*0.5)
        #print('fenv', fenv)
        fenv = np.floor(fenv*100)/100
        actions_env[4] = fenv
        #print('fenvbounded', fenv)
                
        #print('actions', actions)
        #print('actions_env', actions_env)
        return actions, actions_env, dirichlet_distribution.get_concentration()
    
    # Choose an action given some observations
    # single is used to specify if it's online training
    def choose_action(self, observation: npt.NDArray[np.float64], nb_actions: int, single=True):
        try:
              # add batch dimension to the observation if only a single example was provided
              observation = np.expand_dims(observation, axis=0) if single else observation
              # feed the observations through the model to predict the mean and log sigma of each action
              distributions_params = self.model(observation)
              logcon, mus, logsigmas = tf.split(distributions_params, [3, 3, 3], axis=1)
              #print('logcon', logcon)
              #print('SAMPLED Mus', mus)
              mussigmoid = tf.sigmoid(mus) #conversion
              #print('SIGMOID mus', mussigmoid)
              max_std = 0.3
              min_std = 0.005
              logsigmassigmoid = max_std*tf.sigmoid(logsigmas) + min_std #conversion
              #print('LOGSIGMASSIGMOID', logsigmassigmoid)
              actions, actions_env, concentration = self.bound_and_sample_actions(logcon, mussigmoid, logsigmassigmoid, observation[0][0])
              nn_outputs = np.concatenate((concentration.flatten(), mussigmoid.numpy().flatten(), logsigmassigmoid.numpy().flatten()))

              return actions, observation.flatten(), actions_env, nn_outputs
        except Exception as e:
              print('Exception handled in choose_action', e)   

