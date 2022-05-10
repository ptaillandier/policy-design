import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import Sequential
import tensorflow_probability as tfp
import utils

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
    def bound_and_sample_actions(self, mussigmoid, logsigmassigmoid, budget):
        #print('Calling bound and sample actions')
        #print('budget', budget)
        #print('mussigmoid',mussigmoid)
        #Params
        fraction_new_adopt_h = 0.05 #We consider 5 new adopters for the economy help
        nindividuals = 100 #The number of individuals of the simulation is fixed to 100
        actions = np.full(5,-1.0, dtype='float32')
        bounds = np.full(5,1.0, dtype='float32')
        remaining_budget = budget
        #Permute on the order of the 3 dimensions (Economy, Management, Environment)
        dimensions_order = np.random.permutation(3)
        #print('dimensions_order', dimensions_order)
        for dimension in dimensions_order:
            #print('dimension', dimension)
            if dimension == 0:
                #print('Processing economy dimension')
                #print('\t Remaining budget', remaining_budget)
                max_bound = remaining_budget/(fraction_new_adopt_h*nindividuals)
                #print('\t Max bound due to budget', max_bound)
                max_bound = np.minimum(1.0, max_bound)
                #print('\t Max bound due to budget and <=1)', max_bound)
                max_bound = np.maximum(0.0001, max_bound)
                #print('Max bound due to budget and <=1 and >=0.0001)', max_bound)
                max_bound = max_bound.astype(np.float32)
                bounds[0] = max_bound
                #print('\t mussigmoid[0][0])', mussigmoid[0][0])
                #print('\t logsigmassigmoid[0][0]', logsigmassigmoid[0][0])
                #Create distribution
                distribution = tfp.distributions.TruncatedNormal(mussigmoid[0][0], logsigmassigmoid[0][0], low=[0], high=[max_bound])
                print('truncatedNormal_thetaeco_'+str(mussigmoid[0][0].numpy())+'_'+str(logsigmassigmoid[0][0].numpy())+'_'+str(max_bound)+'.png')
                utils.save_plot_distribution(distribution, 'truncatedNormal_thetaeco_'+str(mussigmoid[0][0].numpy())+'_'+str(logsigmassigmoid[0][0].numpy())+'_'+str(max_bound)+'.png')
                action= distribution.sample()
                print('\t thetaeconomy', action)
                print('\t prob(action)', distribution.prob(action), 'logprob(action)', distribution.log_prob(action))
                normalmu = tf.multiply(mussigmoid[0][0], max_bound)
                distribution = tfp.distributions.Normal(normalmu, logsigmassigmoid[0][0])
                print('\t normalthetaeconomy', action)
                print('\t normalprob(action)', tf.exp(distribution.log_prob(action)), 'logprob(action)', distribution.log_prob(action))
                print('Normal_thetaeco_'+str(normalmu.numpy())+'_'+str(logsigmassigmoid[0][0].numpy())+'.png')
                utils.save_plot_distribution(distribution, 'Normal_thetaeco_'+str(normalmu.numpy())+'_'+str(logsigmassigmoid[0][0].numpy())+'.png')
                
                actions[0] = action.numpy().flatten()
                #Update remaining budget
                remaining_budget =  remaining_budget - actions[0]*fraction_new_adopt_h*nindividuals
                #print('\t remaining budget after action', remaining_budget)
            if dimension == 1:
                #print('Processing managing dimension')
                subdimensions_order = np.random.permutation(2)
                for subdimension in subdimensions_order:
                    #print('subdimension', subdimension)
                    if subdimension == 0:
                        #print('Processing management theta')
                        #print('\t Remaining budget', remaining_budget)
                        fraction = actions[2]
                        if fraction ==-1: #First subdimension
                           fraction = np.around(mussigmoid[0][2].numpy().flatten(), decimals=1)
                        #print('\t fraction', fraction)
                        if fraction > 0:
                            max_bound = remaining_budget/(fraction*nindividuals)
                        else:
                            max_bound = 0
                        #print('\t Max bound due to budget', max_bound)
                        max_bound = np.minimum(1.0, max_bound)
                        #print('Max bound due to budget and <=1)', max_bound)
                        max_bound = np.maximum(0.0001, max_bound)
                        #print('Max bound due to budget and <=1 and >=0.0001)', max_bound)

                        max_bound = max_bound.astype(np.float32)
                        bounds[1] = max_bound
                        #print('\t mussigmoid[0][1])', mussigmoid[0][1])
                        #print('\t logsigmassigmoid[0][1]', logsigmassigmoid[0][1])
                        #Create distribution
                        distribution = tfp.distributions.TruncatedNormal(mussigmoid[0][1], logsigmassigmoid[0][1], low=[0], high=[max_bound])
                        action= distribution.sample()
                        #print('\t thetamanagement', action)
                        #print('\t prob(action)', tf.exp(distribution.log_prob(action)), 'logprob(action)', distribution.log_prob(action))
                        actions[1] = action.numpy().flatten()

                    else:
                        #print('Processing fraction management')
                        #print('\t Remaining budget', remaining_budget)
                        theta = actions[1]
                        if theta ==-1: #First subdimension
                           theta = mussigmoid[0][1].numpy().flatten()
                        #print('\t theta', theta)
                        max_bound = np.floor((remaining_budget/(theta*nindividuals))*100)/100
                        #print('\t Max bound due to budget', max_bound)
                        max_bound = np.minimum(1.0, max_bound)
                        #print('Max bound due to budget and <=1)', max_bound)
                        max_bound = np.maximum(0.0001, max_bound)
                        #print('Max bound due to budget and <=1 and >=0.0001)', max_bound)

                        max_bound = max_bound.astype(np.float32)
                        bounds[2] = max_bound
                        #print('\t mussigmoid[0][2])', mussigmoid[0][2])
                        #print('\t logsigmassigmoid[0][2]', logsigmassigmoid[0][2])
                        #Create distribution
                        distribution = tfp.distributions.TruncatedNormal(mussigmoid[0][2], logsigmassigmoid[0][2], low=[0], high=[max_bound])
                        action= distribution.sample()
                        #print('\t thetamanagement', action)
                        #print('\t prob(action)', tf.exp(distribution.log_prob(action)), 'logprob(action)', distribution.log_prob(action))
                        actions[2] = np.floor(action.numpy().flatten()*100)/100

                #Update remaining budget once processed all subdimensions with selected actions
                remaining_budget =  remaining_budget - actions[1]*actions[2]*nindividuals
                #print('remaining budget after management actions', remaining_budget)

            if dimension == 2:
                #print('Processing environment dimension')
                subdimensions_order = np.random.permutation(2)
                for subdimension in subdimensions_order:
                    #print('subdimension', subdimension)
                    if subdimension == 0:
                        #print('Processing environment theta')
                        #print('\t Remaining budget', remaining_budget)
                        fraction = actions[4]
                        if fraction ==-1: #First subdimension
                           fraction = np.around(mussigmoid[0][4].numpy().flatten(), decimals=1)
                        #print('\t fraction', fraction)
                        if fraction > 0:
                            max_bound = remaining_budget/(fraction*nindividuals)
                        else:
                            max_bound = 0
                        #print('\t Max bound due to budget', max_bound)
                        max_bound = np.minimum(1.0, max_bound)
                        #print('Max bound due to budget and <=1)', max_bound)
                        max_bound = np.maximum(0.0001, max_bound)
                        #print('Max bound due to budget and <=1 and >=0.0001)', max_bound)

                        max_bound = max_bound.astype(np.float32)
                        bounds[3] = max_bound
                        #print('\t mussigmoid[0][1])', mussigmoid[0][3])
                        #print('\t logsigmassigmoid[0][1]', logsigmassigmoid[0][3])
                        #Create distribution
                        distribution = tfp.distributions.TruncatedNormal(mussigmoid[0][3], logsigmassigmoid[0][3], low=[0], high=[max_bound])
                        action= distribution.sample()
                        #print('\t thetamanagement', action)
                        #print('\t prob(action)', tf.exp(distribution.log_prob(action)), 'logprob(action)', distribution.log_prob(action))
                        actions[3] = action.numpy().flatten()
                    else:
                        #print('Processing fraction nenvironmental')
                        #print('\t Remaining budget', remaining_budget)
                        theta = actions[3]
                        if theta ==-1: #First subdimension
                           theta = mussigmoid[0][3].numpy().flatten()
                        #print('\t theta', theta)
                        max_bound = np.floor((remaining_budget/(0.5*theta*nindividuals))*100)/100
                        #print('\t Max bound due to budget', max_bound)
                        max_bound = np.minimum(1.0, max_bound)
                        #print('Max bound due to budget and <=1)', max_bound)
                        max_bound = np.maximum(0.0001, max_bound)
                        #print('Max bound due to budget and <=1 and >=0.0001)', max_bound)

                        max_bound = max_bound.astype(np.float32)
                        bounds[4] = max_bound
                        #print('\t mussigmoid[0][2])', mussigmoid[0][4])
                        #print('\t logsigmassigmoid[0][2]', logsigmassigmoid[0][4])
                        #Create distribution
                        distribution = tfp.distributions.TruncatedNormal(mussigmoid[0][4], logsigmassigmoid[0][4], low=[0], high=[max_bound])
                        action= distribution.sample()
                        #print('\t thetamanagement', action)
                        #print('\t prob(action)', tf.exp(distribution.log_prob(action)), 'logprob(action)', distribution.log_prob(action))
                        actions[4] = np.floor(action.numpy().flatten()*100)/100

                    #Update remaining budget once processed all subdimensions with selected actions
                    remaining_budget =  remaining_budget - 0.5*actions[3]*actions[4]*nindividuals
                    #print('remaining budget after environmental actions', remaining_budget)

        return actions, bounds
    
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
              print('SAMPLED Mus', mus)
              mussigmoid = tf.sigmoid(mus) #conversion
              print('SIGMOID mus', mussigmoid)
              max_std = 0.3
              min_std = 0.005
              logsigmassigmoid = max_std*tf.sigmoid(logsigmas) + min_std #conversion
              print('LOGSIGMASSIGMOID', logsigmassigmoid)
              print('before')
              actions, bounds = self.bound_and_sample_actions(mussigmoid, logsigmassigmoid, observation[0][0])
              #print('BOUNDED_ACTIONS', actions)
              print('BOUNDS', bounds)
              #distributions = tfp.distributions.TruncatedNormal(mussigmoid, logsigmassigmoid, low=[0], high=[1])
              #actions = distributions.sample()
              #actions = actions.numpy().flatten()
              #Implement the bounding of actions to not exceed the budget (observation[0])
              #print('actions (Thetaeconomy ,Thetamanagement,Fmanagement,Thetaenvironment ,Fenvironment)', actions)
              #print('remaining budget', observation[0][0])
              #remaining_budget = observation[0][0]
              #nindividuals = 100 #The number of individuals of the simulation is fixed to 100
              
              #Compute how much budget we will be using in management and environmental actions with current actions
              #budget_4_management = nindividuals*actions[2]*actions[1]
              #print('budget_4_management', budget_4_management)
              #budget_4_environmental = 0.5*nindividuals*actions[4]*actions[3]
              #print('budget_4_environmental', budget_4_environmental)
              #We leave budget for a 10 new adopters to get, if enough budget 
              #fraction_new_adopt_h = 0.05
              #budget_4_economy = nindividuals*fraction_new_adopt_h*actions[0]
              #print('budget_4_economy', budget_4_economy)
              #all_budget = budget_4_management + budget_4_environmental + budget_4_economy
              #budget_2_reduce = all_budget - remaining_budget
              
              
              #if budget_2_reduce > 0:
                  #budget_2_reduce_management = budget_2_reduce * (budget_4_management/all_budget)
                  #print('budget_2_reduce_management', budget_2_reduce_management)
                  #budget_2_reduce_environmental = budget_2_reduce * (budget_4_environmental/all_budget)
                  #print('budget_2_reduce_environmental', budget_2_reduce_environmental)
                  #budget_2_reduce_economy = budget_2_reduce * (budget_4_economy/all_budget)
                  #print('budget_2_reduce_economy', budget_2_reduce_economy)

                  #factor_2_reduce_management = (budget_4_management-budget_2_reduce_management)/budget_4_management
                  #print('factor_2_reduce_management', factor_2_reduce_management)
                  #Update management action
                  #actions[2] = actions[2]*factor_2_reduce_management
                  #factor_2_reduce_environmental = (budget_4_environmental-budget_2_reduce_environmental)/budget_4_environmental
                  #print('factor_2_reduce_environmental', factor_2_reduce_environmental)
                  #Update environmenal action
                  #actions[4] = actions[4]*factor_2_reduce_environmental
                  #factor_2_reduce_economy = (budget_4_economy-budget_2_reduce_economy)/budget_4_economy
                  #print('factor_2_reduce_economy', factor_2_reduce_economy)
                  #Update economy action
                  #actions[0] = actions[0]*factor_2_reduce_economy

                  #Recompute the new budget for the new actions
                  #budget_4_management = nindividuals*actions[2]*actions[1]
                  #budget_4_environmental = 0.5*nindividuals*actions[4]*actions[3]
                  #budget_4_economy = nindividuals*fraction_new_adopt_h*actions[0]
                  #print('new budget', budget_4_management+budget_4_environmental+budget_4_economy)
                  #print('actions (Thetaeconomy ,Thetamanagement,Fmanagement,Thetaenvironment ,Fenvironment)', actions)
                  #Check that the actions are not negative
                  #actions = np.maximum(0, actions)
              return actions, observation.flatten(), bounds
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
