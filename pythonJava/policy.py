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
              actions = actions.numpy().flatten()
              #Implement the bounding of actions to not exceed the budget (observation[0])
              #print('actions (Thetaeconomy ,Thetamanagement,Fmanagement,Thetaenvironment ,Fenvironment)', actions)
              #print('remaining budget', observation[0][0])
              remaining_budget = observation[0][0]
              nindividuals = 100 #The number of individuals of the simulation is fixed to 100
              
              #Compute how much budget we will be using in management and environmental actions with current actions
              budget_4_management = nindividuals*actions[2]*actions[1]
              #print('budget_4_management', budget_4_management)
              budget_4_environmental = 0.5*nindividuals*actions[4]*actions[3]
              #print('budget_4_environmental', budget_4_environmental)
              #We leave budget for a 10 new adopters to get, if enough budget 
              fraction_new_adopt_h = 0.1
              budget_4_economy = nindividuals*fraction_new_adopt_h*actions[0]
              #print('budget_4_economy', budget_4_economy)
              all_budget = budget_4_management + budget_4_environmental + budget_4_economy
              budget_2_reduce = all_budget - remaining_budget
              
              
              if budget_2_reduce > 0:
                  budget_2_reduce_management = budget_2_reduce * (budget_4_management/all_budget)
                  #print('budget_2_reduce_management', budget_2_reduce_management)
                  budget_2_reduce_environmental = budget_2_reduce * (budget_4_environmental/all_budget)
                  #print('budget_2_reduce_environmental', budget_2_reduce_environmental)
                  budget_2_reduce_economy = budget_2_reduce * (budget_4_economy/all_budget)
                  #print('budget_2_reduce_economy', budget_2_reduce_economy)

                  factor_2_reduce_management = (budget_4_management-budget_2_reduce_management)/budget_4_management
                  #print('factor_2_reduce_management', factor_2_reduce_management)
                  #Update management action
                  actions[2] = actions[2]*factor_2_reduce_management
                  factor_2_reduce_environmental = (budget_4_environmental-budget_2_reduce_environmental)/budget_4_environmental
                  #print('factor_2_reduce_environmental', factor_2_reduce_environmental)
                  #Update environmenal action
                  actions[4] = actions[4]*factor_2_reduce_environmental
                  factor_2_reduce_economy = (budget_4_economy-budget_2_reduce_economy)/budget_4_economy
                  #print('factor_2_reduce_economy', factor_2_reduce_economy)
                  #Update economy action
                  actions[0] = actions[0]*factor_2_reduce_economy

                  #Recompute the new budget for the new actions
                  budget_4_management = nindividuals*actions[2]*actions[1]
                  budget_4_environmental = 0.5*nindividuals*actions[4]*actions[3]
                  budget_4_economy = nindividuals*fraction_new_adopt_h*actions[0]
                  #print('new budget', budget_4_management+budget_4_environmental+budget_4_economy)
                  #print('actions (Thetaeconomy ,Thetamanagement,Fmanagement,Thetaenvironment ,Fenvironment)', actions)

              return actions, observation.flatten()
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
